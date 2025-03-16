import os
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, g, send_from_directory
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta  # Add this for month calculations
import base64
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import DictCursor
import shutil

app = Flask(__name__, 
            static_folder='website/static', 
            template_folder='website/templates')

app.secret_key = os.urandom(24)

# Load env variables first
load_dotenv()

# Then configure API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

#  upload folder path
UPLOAD_FOLDER = "/tmp/uploads"

# clean uplaod fodler
if os.path.exists(UPLOAD_FOLDER):
    try:
        if os.path.isfile(UPLOAD_FOLDER):
            os.remove(UPLOAD_FOLDER)
        else:
            shutil.rmtree(UPLOAD_FOLDER)
    except Exception as e:
        print(f"Error cleaning up uploads folder: {e}")
        raise

# Create  uploads folder
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
except Exception as e:
    print(f"Error creating uploads directory: {e}")
    raise

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# error handling for missing env variables
if not os.getenv('DATABASE_URL'):
    raise ValueError("DATABASE_URL environment variable is required")
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Update the database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Update upload folder for Vercel
UPLOAD_FOLDER = '/tmp' if os.getenv('VERCEL_ENV') else 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Add error handling for database connection
try:
    DATABASE_URL = os.getenv('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL
    )
except Exception as e:
    print(f"Database connection error: {e}")
    raise

# Initialize database connection pool
db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)

def get_db():
    if 'db' not in g:
        g.db = db_pool.getconn()
        g.db.cursor_factory = DictCursor
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, 'db', None)
    if db is not None:
        db_pool.putconn(db)

def init_db():
    with app.app_context():
        db = get_db()
        try:
            with db.cursor() as cursor:
                with open('src/schema.sql', 'r') as f:
                    cursor.execute(f.read())
            db.commit()
            print("Database initialized successfully!")
        except Exception as e:
            db.rollback()
            print(f"Error initializing database: {e}")

# File handling condtion
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def allowed_file(filename):
    """Check if file type is allowed and size is within limits"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
        
    return True

def validate_file(file):
    """Validate both file type and size"""
    if not file:
        return False, "No file provided"
        
    if not allowed_file(file.filename):
        return False, "File type not allowed"
        
    # Check file size
    file.seek(0, 2) 
    size = file.tell() 
    file.seek(0) 
    
    if size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
        
    return True, "File is valid"

# main class
class BillProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.CHUNK_SIZE = 5 

    def print_processing_summary(self, file_path, file_type, text_length):
        """Print a summary of the processed file."""
        print("\n" + "="*50)
        print(f"PROCESSING SUMMARY FOR: {os.path.basename(file_path)}")
        print("="*50)
        print(f"File Type: {file_type}")
        print(f"Extracted Text Length: {text_length} characters")
        print("-"*50)

    def print_extracted_text_sample(self, text):
        """Print a sample of the extracted text."""
        print("\nEXTRACTED TEXT SAMPLE:")
        print("-"*50)
        if len(text) > 400:
            print("First 200 characters:")
            print(text[:200] + "...")
            print("\nLast 200 characters:")
            print("..." + text[-200:])
        else:
            print(text)
        print("-"*50)

    def print_results(self, results):
        """Print the analyzed results in a formatted way."""
        print("\nANALYSIS RESULTS:")
        print("="*50)
        for i, bill in enumerate(results, 1):
            print(f"\nBill #{i}:")
            print("-"*25)
            print(f"Date: {bill['invoice_date']}")
            print(f"Category: {bill['category']}")
            print(f"Amount: {bill['total_amount']}")
            print(f"Confidence: {bill['confidence']}")
        print("="*50 + "\n")

    def process_pdf(self, pdf_path):
        """Process a PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text_content = ""
            
            for page in reader.pages:
                text_content += page.extract_text() + "\n"
                
            return text_content.strip()
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None

    def process_image(self, image_file):
        """Process image directly with Gemini"""
        try:
            # Read image file
            image_parts = [{
                "mime_type": image_file.content_type,
                "data": image_file.read()
            }]
            
            # prompt for Gemini
            prompt = """Extract text from the following image
            Instructions:
            1. Extract the text from the image.
            2. Return the extracted text as a plain string."""
            
            # Generate content with image and prompt
            response = self.model.generate_content([prompt, *image_parts])
            
            # Get the response text
            if response.text:
                return response.text
            return None

        except Exception as e:
            print(f"Error processing image with Gemini: {e}")
            return None

    def process_text_with_gemini(self, extracted_text):
        """Process the extracted text with Gemini."""
        if not extracted_text.strip():
            print("Error: No text provided for Gemini processing")
            return []

        print("\nPROCESSING WITH GEMINI")
        print("=" * 50)
        print(f"Text length: {len(extracted_text)} characters")

        # Gemini prompt
        prompt = f"""
            Analyze the following bill text. It may come from an image or PDF, and there may be multiple bills. Treat each bill separately.

            {extracted_text}

            Instructions:
            1. Extract the invoice date (any format but). Use the most recent date from text and and consider the fact that we're targetating indian user so most common formate in bill will be dd-mm-yyyy.
            2. Extract the total amount (positive, non-zero). If multiple, choose the most accurate.
            3. Classify the bill into one of the following categories:
            Food (exclude groceries), Groceries, Utilities, Travel, Transportation (fuel included), Shopping, Health, Education, Entertainment, Personal Care (gym too), EMI, Rent, Other.
            4. Confidence level based on clarity: high/medium/low.

            Return only a JSON array with these details:
            [
                {{
                    "invoice_date": "YYYY-MM-DD",
                    "category": "category_from_list",
                    "total_amount": number_without_currency_symbol,
                    "confidence": "high/medium/low"
                }}
            ]

            if you think its not a bill, return "No bill in file" for images/PDFs. For PDFs with some non-bill pages, return null for those pages and state "n page has no bill"
            If category or total_amount is missing return "no bill found" and retrun this json format :-

            {{
                "invoice_date": null,
                "category": null,
                "total_amount": null,
                "confidence": "no bill"
            }}
        """

        print("\nSending to Gemini...")
        response = self.model.generate_content([prompt])
        response_text = response.text.strip()

        print("\nParsing Gemini response...")
        response_text = response_text.replace('```json', '').replace('```', '').strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            print("Error: Invalid JSON response from Gemini")
            return []

        # Assign the bill upload date if invoice_date is null or missing
        upload_date = datetime.now().strftime("%Y-%m-%d")
        for item in result:
            if item.get("invoice_date") in [None, "", "null"]:
                item["invoice_date"] = upload_date

        # Print results directly from Gemini's output
        self.print_results(result)
        return result

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        db = get_db()
        try:
            with db.cursor() as cursor:
                cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    session['user_id'] = user['user_id']
                    session['username'] = user['username']
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid email or password', 'error')
                    return redirect(url_for('login'))
        except Exception as e:
            flash('Login failed', 'error')
            print(f"Login error: {e}")
            return redirect(url_for('login'))

    # GET request - show login form
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']

        db = get_db()
        try:
            with db.cursor() as cursor:
                # Check if email already exists
                cursor.execute("SELECT * FROM Users WHERE email = %s", (email,))
                if cursor.fetchone():
                    flash('Email already registered', 'error')
                    return redirect(url_for('login'))

                # Insert new user
                cursor.execute(
                    "INSERT INTO Users (email, password, username) VALUES (%s, %s, %s) RETURNING user_id",
                    (email, generate_password_hash(password), username)
                )
                user_id = cursor.fetchone()[0]
                db.commit()
                
                session['user_id'] = user_id
                session['username'] = username
                flash('Registration successful!', 'success')
                return redirect(url_for('login'))
            
        except Exception as e:
            db.rollback()
            flash('Registration failed', 'error')
            print(f"Registration error: {e}")
            return redirect(url_for('register'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    current_year = datetime.now().year
    db = get_db()

    with db.cursor() as cursor:
        # Fetch user details
        cursor.execute("""
            SELECT username, email, profile_picture 
            FROM Users WHERE user_id = %s
        """, (session['user_id'],))
        user_data = cursor.fetchone()
        if not user_data:
            flash('User not found', 'error')
            return redirect(url_for('login'))

        # Initialize default values
        yearly_expenses = 0
        yearly_budget = 0
        total_sum = 0
        spent_amount = 0

        # Fetch yearly expenses (with error handling)
        try:
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as yearly_expenses
                FROM Bill 
                WHERE user_id = %s 
                AND EXTRACT(YEAR FROM invoice_date) = %s
            """, (session['user_id'], current_year))
            yearly_expenses = cursor.fetchone()[0] or 0
        except Exception as e:
            print(f"Error fetching yearly expenses: {e}")

        # Fetch yearly budget (with error handling)
        try:
            cursor.execute("""
                SELECT COALESCE(SUM(budget), 0) as yearly_budget
                FROM MonthlyBudget 
                WHERE user_id = %s 
                AND month LIKE %s
            """, (session['user_id'], f"{current_year}%"))
            yearly_budget = cursor.fetchone()[0] or 0
        except Exception as e:
            print(f"Error fetching yearly budget: {e}")

        # Fetch monthly data (with error handling)
        try:
            # Get current month for budget
            current_month = datetime.now().strftime('%Y-%m')
            
            # Simplified query to get current month's budget only
            cursor.execute("""
                SELECT budget
                FROM MonthlyBudget 
                WHERE user_id = %s 
                AND month = %s
            """, (session['user_id'], current_month))
            
            budget_result = cursor.fetchone()
# Ensure no None values are returned
            total_budget = float(budget_result[0]) if budget_result and budget_result[0] is not None else 0.0



            # Keep selected_month for expenses
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as total_sum
                FROM Bill 
                WHERE user_id = %s 
                AND to_char(invoice_date, 'YYYY-MM') = %s
            """, (session['user_id'], selected_month))
            
            expense_result = cursor.fetchone()
            total_sum = round(float(expense_result['total_sum']), 2)
            spent_amount = total_sum  # Update spent amount to match actual expenses
        except Exception as e:
            print(f"Error fetching monthly data: {e}")

        # Calculate percentages and remaining amounts (safely)
        budget_percentage = round((total_sum / total_budget * 100) if total_budget > 0 else 0, 2)
        budget_remaining = round(max(total_budget - total_sum, 0),2)
        yearly_remaining = max(yearly_budget - yearly_expenses, 0)
        budget_yearpercentage = round((yearly_expenses / yearly_budget * 100) if yearly_budget > 0 else 0, 2)

        # Generate month options
        current = datetime.now()
        month_options = [
            ((current - relativedelta(months=i)).strftime('%Y-%m'),
             (current - relativedelta(months=i)).strftime('%b %Y'))
            for i in range(12)
        ]

        # Handle profile picture
        image_data = None
        if user_data.get('profile_picture'):
            try:
                image_data = base64.b64encode(user_data['profile_picture']).decode('utf-8')
            except Exception as e:
                print(f"Error encoding profile picture: {e}")

        # Fetch monthly trends data for stacked bar chart
        current = datetime.now()
        months = []
        expenses = []
        savings = []

        # Get data for last 12 months
        for i in range(12):
            month = (current - relativedelta(months=i)).strftime('%Y-%m')
            months.append((current - relativedelta(months=i)).strftime('%b %Y'))
            
            # Get expenses for this month
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as total_expense
                FROM Bill 
                WHERE user_id = %s 
                AND to_char(invoice_date, 'YYYY-MM') = %s
            """, (session['user_id'], month))
            total_expense = float(cursor.fetchone()['total_expense'])
            expenses.append(total_expense)
            
            # Get budget and calculate savings
            cursor.execute("""
                SELECT budget 
                FROM MonthlyBudget 
                WHERE user_id = %s AND month = %s
            """, (session['user_id'], month))
            budget_info = cursor.fetchone()
            total_budget = float(budget_info['budget']) if budget_info else 0
            savings.append(max(total_budget - total_expense, 0))

        # Reverse lists to show oldest to newest
        months.reverse()
        expenses.reverse()
        savings.reverse()

        return render_template(
            'dashboard.html',
            username=user_data['username'],
            email=user_data['email'],
            total_sum=total_sum,
            spent_amount=spent_amount,
            budget_percentage=budget_percentage,
            budget_remaining=budget_remaining,
            yearly_remaining=yearly_remaining,
            yearly_expenses=yearly_expenses,
            yearly_budget=yearly_budget,
            budget_yearpercentage=budget_yearpercentage,
            current_month=selected_month,
            month_options=month_options,
            image_data=image_data,
            monthly_expense_total=total_sum,
            months=months,
            expenses=expenses,
            total_budget=total_budget,
            savings=savings
            
        )

@app.route('/upload_bill', methods=['POST'])
def upload_bill():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if 'bill_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['bill_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))

    # Validate file
    is_valid, message = validate_file(file)
    if not is_valid:
        flash(message, 'danger')
        return redirect(url_for('dashboard'))

    try:
        bill_processor = BillProcessor()
        
        if file.filename.lower().endswith('.pdf'):
            # For PDFs, save temporarily to extract text
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(temp_path)
            try:
                extracted_text = bill_processor.process_pdf(temp_path)
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            # For images, send directly to Gemini
            extracted_text = bill_processor.process_image(file)

        if not extracted_text:
            flash('Could not extract text from file', 'danger')
            return redirect(url_for('dashboard'))

        # Process extracted text
        results = bill_processor.process_text_with_gemini(extracted_text)
        
        if not results:
            flash('No bill information could be extracted', 'danger')
            return redirect(url_for('dashboard'))

        # Store results in database
        db = get_db()
        with db.cursor() as cursor:
            for bill in results:
                if bill.get('total_amount') and bill.get('category'):
                    cursor.execute("""
                        INSERT INTO Bill 
                        (user_id, total_amount, category, invoice_date, confidence_level)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING bill_id
                    """, (
                        session['user_id'],
                        float(bill['total_amount']),
                        bill['category'],
                        bill['invoice_date'],
                        bill['confidence']
                    ))
            db.commit()
            flash('Bill processed and stored successfully!', 'success')

    except Exception as e:
        print(f"Error processing bill: {e}")
        if 'db' in locals():
            db.rollback()
        flash(f'Error processing file: {str(e)}', 'danger')
        
    return redirect(url_for('dashboard'))

@app.route('/get_expense_breakdown', methods=['GET'])
def get_expense_breakdown():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    
    try:
        db = get_db()
        with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute("""
                WITH monthly_totals AS (
                    SELECT SUM(total_amount) as month_total
                    FROM Bill 
                    WHERE user_id = %s 
                    AND date_trunc('month', invoice_date)::date = to_date(%s, 'YYYY-MM')
                )
                SELECT 
                    category,
                    COALESCE(SUM(total_amount), 0) as total_amount,
                    (SELECT month_total FROM monthly_totals) as month_total
                FROM Bill 
                WHERE user_id = %s 
                AND date_trunc('month', invoice_date)::date = to_date(%s, 'YYYY-MM')
                GROUP BY category
            """, (session['user_id'], selected_month, session['user_id'], selected_month))
            
            expenses = cursor.fetchall()
            
            if not expenses:
                return jsonify({
                    'category': [],
                    'total_amount': [],
                    'month_total': 0
                })

            categories = [row['category'] for row in expenses]
            amounts = [float(row['total_amount']) for row in expenses]
            month_total = float(expenses[0]['month_total'] or 0)


            return jsonify({
                'category': categories,
                'total_amount': amounts,
                'month_total': round(month_total, 2)
            })

    except Exception as e:
        print(f"Error fetching expense breakdown: {e}")
        return jsonify({
            'category': [],
            'total_amount': [],
            'month_total': 0
        })

@app.route('/get_recent_bills', methods=['GET'])
def get_recent_bills():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    
    try:
        db = get_db()
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT invoice_date::date, category, total_amount 
                FROM Bill 
                WHERE user_id = %s 
                AND to_char(invoice_date, 'YYYY-MM') = %s
                ORDER BY invoice_date DESC
            """, (session['user_id'], selected_month))
            
            bills = cursor.fetchall()
            
            bill_list = [{
                'date': bill['invoice_date'].strftime('%Y-%m-%d'),
                'category': bill['category'],
                'amount': float(bill['total_amount'])
            } for bill in bills]
            
            return jsonify({'bills': bill_list})
            
    except Exception as e:
        print(f"Error fetching recent bills: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_budget', methods=['POST'])
def set_budget():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        budget = float(request.form['budget'])
        current_month = datetime.now().strftime('%Y-%m')

        if budget <= 0:
            flash('Budget must be positive', 'error')
            return redirect(url_for('dashboard'))

        db = get_db()
        with db.cursor() as cursor:
            cursor.execute("""
                INSERT INTO MonthlyBudget (user_id, budget, month, spent)
                VALUES (%s, %s, %s, 0)
                ON CONFLICT (user_id, month) 
                DO UPDATE SET budget = EXCLUDED.budget
                RETURNING budget
            """, (session['user_id'], budget, current_month))
            db.commit()
            
        flash('Budget set successfully!', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        print(f"Error setting budget: {e}")
        flash('Error setting budget', 'error')
        return redirect(url_for('dashboard'))

@app.route('/get_monthly_expense_total', methods=['GET'])
def get_monthly_expense_total():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))

    db = get_db()
    with db.cursor() as cursor:
        cursor.execute("""
            SELECT COALESCE(SUM(total_amount), 0) as total
            FROM Bill
            WHERE user_id = %s
            AND to_char(invoice_date, 'YYYY-MM') = %s
        """, (session['user_id'], selected_month))
        result = cursor.fetchone()
        total = float(result['total']) if result['total'] else 0.0

    return jsonify({'total_expense': total})

@app.route('/get_monthly_budget', methods=['GET'])
def get_monthly_budget():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    curr_month = datetime.now().strftime('%Y-%m')
    
    try: 
        db = get_db()
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT COALESCE(budget, 0) as budget
                FROM MonthlyBudget
                WHERE user_id = %s AND month = %s
            """, (session['user_id'], curr_month))
            result = cursor.fetchone()
               
            curr_budget = float(result['budget']) if result and result['budget'] is not None else 0.0
    
            response_data = {'monthly_budget': curr_budget}
            
            return jsonify(response_data)
    
    except Exception as e:
        print(f"Error in get_monthly_budget: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_budget_for_selectmonth', methods=['GET'])
def get_budget_for_selectmonth():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))

    try:
        db = get_db()
        with db.cursor() as cursor:  # Create a cursor
            cursor.execute("""
                SELECT budget 
                FROM MonthlyBudget 
                WHERE user_id = %s AND month = %s
            """, (session['user_id'], selected_month))
            budget_info = cursor.fetchone()

            if not budget_info:
                return jsonify({
                    'monthly-budget': 0
                })

            return jsonify({
                'monthly-budget': round(float(budget_info['budget']), 2)
            })

    except Exception as e:
        print(f"Error fetching monthly budget: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        # Get form data
        name = request.form.get('name')
        profile_picture = request.files.get('profile_picture')
        
        # Get database connection from pool
        db = get_db()
        
        with db.cursor() as cursor:
            # Initialize query parts and parameters
            query_parts = []
            params = []

            # Add name update if provided
            if name and name.strip():
                query_parts.append("username = %s")
                params.append(name.strip())

            # Handle pfp updat 
            if profile_picture and profile_picture.filename:
                if not allowed_file(profile_picture.filename):
                    flash('Please upload a valid image file (JPG, PNG, or JPEG).', 'error')
                    return redirect(url_for('dashboard'))
                
                try:
                    # Read and process image data
                    image_data = profile_picture.read()
                    query_parts.append("profile_picture = %s")
                    params.append(image_data)
                except Exception as e:
                    print(f"Error processing image: {e}")
                    flash('Error processing image file.', 'error')
                    return redirect(url_for('dashboard'))

            # Only proceed if user input something to update
            if query_parts:
                query = "UPDATE Users SET " + ", ".join(query_parts)
                query += " WHERE user_id = %s"
                params.append(session['user_id'])

                cursor.execute(query, tuple(params))
                db.commit()

                # Update session data if user change name
                if name and name.strip():
                    session['username'] = name.strip()

                flash('Profile updated successfully!', 'success')
            else:
                flash('No changes to update.', 'info')

    except Exception as e:
        print(f"Profile update error: {e}")
        db.rollback()
        flash('An error occurred while updating your profile.', 'error')
    
    return redirect(url_for('dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
    # Clear the session data
    session.pop('user_id', None)
    session.pop('username', None)

    return redirect(url_for('login'))

@app.route('/monthly-summary', methods=['POST'])
def monthly_summary():
    user_id = session.get('user_id')  
    selected_month = request.json.get('month')

    if not user_id or not selected_month:
        return jsonify({"error": "Missing user ID or month"}), 400

    try:
        db = get_db()
        with db.cursor() as cursor:
            cursor.execute("""
                SELECT category, SUM(total_amount) AS total_spent
                FROM Bill
                WHERE user_id = %s AND to_char(invoice_date, 'YYYY-MM') = %s
                GROUP BY category
                ORDER BY total_spent DESC
                LIMIT 1
            """, (user_id, selected_month))
            result = cursor.fetchone()
            
            if result:
                return jsonify({
                    "highest_category": {
                        "category": result['category'],
                        "total_spent": float(result['total_spent'])
                    }
                })
            return jsonify({"highest_category": None})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, timeout=30)