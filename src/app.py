import os
import uuid
import base64
import json
import shutil
import tempfile
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import psycopg2
from psycopg2.pool import ThreadedConnectionPool  # Use ThreadedConnectionPool
from psycopg2.extras import DictCursor
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, g, send_from_directory
import google.generativeai as genai
import logging
from functools import wraps
import timeout_decorator
from PyPDF2 import PdfReader

# Configuration & Initialization ---
app = Flask(__name__, static_folder='website/static', template_folder='website/templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))

@app.template_filter('format_number')
def format_number(value):
    return f"{value:.2f}"

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv() 
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY missing.")
    raise ValueError("GEMINI_API_KEY required")
genai.configure(api_key=GEMINI_API_KEY)

# Railway Configuration
UPLOAD_FOLDER = '/tmp'  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILE_SIZE = 2 * 1024 * 1024
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL missing.")
    raise ValueError("DATABASE_URL required")

#db Connection
db_pool = None

def init_db_pool():
    global db_pool
    if db_pool is None:
        try:
            # Optimized for Supabase transaction pooler
            db_pool = ThreadedConnectionPool(
                minconn=1,    
                maxconn=5,   # Reduced for transaction pooler
                dsn=DATABASE_URL,
                # Remove keepalives for transaction pooler compatibility
                options='-c default_transaction_isolation=read_committed'
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    # Test connection correctly with proper connection handling
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1") 
            cursor.fetchone() 
        logger.info("Database connection tested successfully")
    finally:
        if conn:
            db_pool.putconn(conn)  # Always return the connection

# Initialize database pool lazily - only when first needed
def get_db_pool():
    global db_pool
    if db_pool is None:
        init_db_pool()
    return db_pool


@app.before_request
def before_request():
    if not hasattr(g, 'db'):
        g.db = get_db()

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, 'db', None)
    if db is not None and db_pool is not None:
        try:
            if error:
                db.rollback()
            db_pool.putconn(db)
            logger.debug("Database connection returned to pool.")
        except Exception as e:
            logger.error(f"Error returning connection to pool: {e}")

def get_db():
    if 'db' not in g:
        try:
            pool = get_db_pool()  # Use lazy initialization
            g.db = pool.getconn()
            g.db.cursor_factory = DictCursor
            logger.debug("New database connection acquired from pool.")
        except Exception as e:
            logger.error(f"Failed to get a database connection: {e}")
            raise
    return g.db

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

# file rules
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file(file):
    if not file:
        return False, "No file provided"
    if not allowed_file(file.filename):
        return False, "File type not allowed"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE / 1024 / 1024}MB"

    return True, "File is valid"

# Bill Processing
class BillProcessor:
    def __init__(self):
        # Use gemini-2.0-flash-lite for better efficiency and lower cost
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

    def process_pdf(self, pdf_path):
        try:
            logger.info(f"Processing PDF: {os.path.basename(pdf_path)}")
            with open(pdf_path, 'rb') as file:
                pdf_data = file.read()
            
            return self.extract_bill_data(pdf_data, "application/pdf")
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return None

    def process_image(self, image_file):
        try:
            logger.info(f"Processing image: {image_file.filename}")
            # No need to read here - file was already rewound in upload_bill
            return self.extract_bill_data(image_file.read(), image_file.content_type)
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None
    
    def extract_bill_data(self, file_data, content_type):
        try:
            image_parts = [{"mime_type": content_type, "data": file_data}]
            
            prompt = """
                Analyze the content of the provided file.
                If it is an image, assume it contains one bill.
                If it is a PDF, it may contain multiple bills. Treat each bill or distinct page/section separately.

                For each bill or distinct page/section extract the following:-

                1. Extract the invoice date. Use the most recent date if multiple. Output format: YYYY-MM-DD.
                2. Extract the total amount (positive, non-zero). If multiple, choose the most accurate.
                3. Classify the bill into one of these categories: Food (prepared), Groceries, Utilities, Travel, Transportation (fuel incl), Shopping, Health, Education, Entertainment, Personal Care, EMI, Rent, Other.
                4. Confidence level based on clarity: high/medium/low.

                Output: Return a JSON array.

                If for a bill/section: Category (3) AND Total (2) are valid: Add { "invoice_date": "YYYY-MM-DD", "category": "category_from_list", "total_amount": number_without_currency_symbol, "confidence": "high/medium/low" } to array.

                Else (Category (3) OR Total (2) missing/invalid, OR section has no bill): Add { "invoice_date": null, "category": null, "total_amount": null, "confidence": "no bill" } to array.

                If entire content has no segments: Return [].
            """
            
            logger.info("Sending file to Gemini for direct processing")
            response = self.model.generate_content([prompt, *image_parts])
            response_text = response.text.strip()
            logger.info(f"Gemini raw response: {response_text}")
            
            # Clean and parse the response
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            logger.info(f"Cleaned response: {response_text}")
            
            try:
                result = json.loads(response_text)
                logger.info(f"Parsed JSON result: {json.dumps(result, indent=2)}")
                
                # Handle default date if needed
                upload_date = datetime.now().strftime("%Y-%m-%d")
                for item in result:
                    if isinstance(item, dict) and item.get("invoice_date") in [None, "", "null"]:
                        item["invoice_date"] = upload_date
                
                # Filter out invalid bills
                valid_bills = [bill for bill in result if isinstance(bill, dict) and 
                              bill.get("total_amount") is not None and
                              bill.get("category") is not None and
                              bill.get("confidence") != "no bill"]
                
                logger.info(f"Found {len(valid_bills)} valid bills")
                return valid_bills
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from Gemini: {response_text}, Error: {e}")
                return []
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return []

# Route Decorators
def handle_timeout(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return timeout_decorator.timeout(30)(func)(*args, **kwargs)
        except timeout_decorator.TimeoutError:
            logger.error(f"Function {func.__name__} timed out")
            return jsonify({'error': 'Request timed out'}), 504
    return wrapper

# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Clear any success messages when accessing login page
    if request.method == 'GET':
        session.pop('_flashes', None)  # Clear all flash messages on GET

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            db = get_db()
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
            logger.error(f"Login error: {e}")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        username = request.form.get('username', '').strip()

        # Validate required fields
        if not email or not password or not username:
            flash('All fields are required', 'error')
            return redirect(url_for('register'))

        # Basic email validation
        if '@' not in email or '.' not in email:
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('register'))

        # Password validation
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'error')
            return redirect(url_for('register'))

        try:
            db = get_db()
            with db.cursor() as cursor:
                # Check for existing email
                cursor.execute("SELECT user_id FROM Users WHERE email = %s", (email,))
                if cursor.fetchone():
                    flash('Email already registered', 'error')
                    return redirect(url_for('register'))

                # Insert new user
                cursor.execute(
                    "INSERT INTO Users (email, password, username) VALUES (%s, %s, %s) RETURNING user_id",
                    (email, generate_password_hash(password), username)
                )
                user_id = cursor.fetchone()[0]
                db.commit()

                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))

        except psycopg2.IntegrityError as e:
            db.rollback()
            logger.error(f"Database integrity error: {e}")
            flash('Registration failed - please try again', 'error')
            return redirect(url_for('register'))
        except Exception as e:
            db.rollback()
            logger.error(f"Registration error: {e}")
            flash('An unexpected error occurred', 'error')
            return redirect(url_for('register'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))
    current_year = datetime.now().year

    try:
        db = get_db()
        with db.cursor() as cursor:
            # user details
            cursor.execute("""
                SELECT username, email, profile_picture
                FROM Users WHERE user_id = %s
            """, (session['user_id'],))
            user_data = cursor.fetchone()
            if not user_data:
                flash('User not found', 'error')
                return redirect(url_for('login'))

            # init default values
            yearly_expenses = 0
            yearly_budget = 0
            total_sum = 0
            spent_amount = 0

            #  yearly expenses
            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as yearly_expenses
                FROM Bill
                WHERE user_id = %s
                AND EXTRACT(YEAR FROM invoice_date) = %s
            """, (session['user_id'], current_year))
            yearly_expenses = cursor.fetchone()[0] or 0

            #  yearly budget
            cursor.execute("""
                SELECT COALESCE(SUM(budget), 0) as yearly_budget
                FROM MonthlyBudget
                WHERE user_id = %s
                AND month LIKE %s
            """, (session['user_id'], f"{current_year}%"))
            yearly_budget = cursor.fetchone()[0] or 0

            #  monthly data
            current_month = datetime.now().strftime('%Y-%m')
            cursor.execute("""
                SELECT budget
                FROM MonthlyBudget
                WHERE user_id = %s
                AND month = %s
            """, (session['user_id'], current_month))
            budget_result = cursor.fetchone()
            total_budget = float(budget_result[0]) if budget_result and budget_result[0] is not None else 0.0

            cursor.execute("""
                SELECT COALESCE(SUM(total_amount), 0) as total_sum
                FROM Bill
                WHERE user_id = %s
                AND to_char(invoice_date, 'YYYY-MM') = %s
            """, (session['user_id'], selected_month))

            expense_result = cursor.fetchone()
            total_sum = round(float(expense_result['total_sum']), 2)
            spent_amount = total_sum

            # Calculate percentages and remaining amounts
            budget_percentage = round((total_sum / total_budget * 100) if total_budget > 0 else 0, 2)
            budget_remaining = round(max(total_budget - total_sum, 0), 2)
            yearly_remaining = max(yearly_budget - yearly_expenses, 0)
            budget_yearpercentage = round((yearly_expenses / yearly_budget * 100) if yearly_budget > 0 else 0, 2)

            # Generate month options
            current = datetime.now()
            month_options = [
                ((current - relativedelta(months=i)).strftime('%Y-%m'),
                 (current - relativedelta(months=i)).strftime('%b %Y'))
                for i in range(12)
            ]

            # Handle pfp
            image_data = None
            if user_data.get('profile_picture'):
                try:
                    image_data = base64.b64encode(user_data['profile_picture']).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encoding profile picture: {e}")

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

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard data', 'error')
        return redirect(url_for('login'))

@app.route('/upload_bill', methods=['POST'])
def upload_bill():
    if 'user_id' not in session:
        flash('Please log in to continue', 'info')
        return redirect(url_for('login'))

    if 'bill_file' not in request.files:
        flash('No file selected', 'info')
        return redirect(url_for('dashboard'))

    file = request.files['bill_file']
    if file.filename == '':
        flash('Please select a file', 'info')
        return redirect(url_for('dashboard'))

    is_valid, message = validate_file(file)
    if not is_valid:
        flash(message, 'info')
        return redirect(url_for('dashboard'))

    try:
        logger.info(f"Processing file: {file.filename}, type: {file.content_type}")
        bill_processor = BillProcessor()
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            file.save(tmp_file.name)
            logger.info(f"File saved to temporary location: {tmp_file.name}")
            
            # Process file and get bill data directly
            if file.filename.lower().endswith('.pdf'):
                logger.info("Processing as PDF file")
                results = bill_processor.process_pdf(tmp_file.name)
            else:
                logger.info("Processing as image file")
                file.seek(0) 
                results = bill_processor.process_image(file)
            
            os.unlink(tmp_file.name)
            logger.info("Temporary file deleted")

            if not results:
                flash('No bill information could be extracted', 'info')
                logger.warning("No bill information returned from Gemini")
                return redirect(url_for('dashboard'))

            logger.info(f"Processing {len(results)} bills")
            
            db = get_db()
            try:
                with db.cursor() as cursor:
                    for i, bill in enumerate(results):
                        if bill.get('total_amount') and bill.get('category'):
                            logger.info(f"Saving bill {i+1}: {json.dumps(bill)}")
                            cursor.execute("""
                                INSERT INTO Bill (user_id, total_amount, category, invoice_date, confidence_level)
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                session['user_id'],
                                float(bill['total_amount']),
                                bill['category'],
                                bill['invoice_date'],
                                bill['confidence']
                            ))
                        else:
                            logger.warning(f"Skipping bill {i+1} due to missing required fields: {json.dumps(bill)}")
                db.commit()
                logger.info("All bills saved to database successfully")

                # Add notification about the month of the uploaded bill
                if results and len(results) > 0 and 'invoice_date' in results[0]:
                    bill_month_date = datetime.strptime(results[0]['invoice_date'], '%Y-%m-%d')
                    bill_month_name = bill_month_date.strftime('%B %Y')
                    flash(f'Bill processed successfully for {bill_month_name}!', 'success')
                else:
                    flash('Bill processed successfully!', 'success')

                return redirect(url_for('dashboard'))
            except Exception as db_error:
                db.rollback()
                logger.error(f"Database error: {db_error}")
                flash('Unable to save bill data', 'info')

    except Exception as e:
        logger.error(f"Bill processing error: {e}", exc_info=True)
        flash('Unable to process bill', 'info')

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
        logger.error(f"Error fetching expense breakdown: {e}")
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
        logger.error(f"Error fetching recent bills: {e}")
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
        logger.error(f"Error setting budget: {e}")
        flash('Error setting budget', 'error')
        return redirect(url_for('dashboard'))

@app.route('/get_monthly_expense_total', methods=['GET'])
def get_monthly_expense_total():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))

    try:
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

    except Exception as e:
        logger.error(f"Error in get_monthly_expense_total: {e}")
        return jsonify({'error': str(e)}), 500

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
        logger.error(f"Error in get_monthly_budget: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_budget_for_selectmonth', methods=['GET'])
def get_budget_for_selectmonth():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    selected_month = request.args.get('month', datetime.now().strftime('%Y-%m'))

    try:
        db = get_db()
        with db.cursor() as cursor:
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
        logger.error(f"Error fetching monthly budget: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        name = request.form.get('name')
        profile_picture = request.files.get('profile_picture')

        db = get_db()

        with db.cursor() as cursor:
            query_parts = []
            params = []

            if name and name.strip():
                query_parts.append("username = %s")
                params.append(name.strip())

            if profile_picture and profile_picture.filename:
                if not allowed_file(profile_picture.filename):
                    flash('Please upload a valid image file (JPG, PNG, or JPEG).', 'error')
                    return redirect(url_for('dashboard'))

                try:
                    image_data = profile_picture.read()
                    query_parts.append("profile_picture = %s")
                    params.append(image_data)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    flash('Error processing image file.', 'error')
                    return redirect(url_for('dashboard'))

            if query_parts:
                query = "UPDATE Users SET " + ", ".join(query_parts)
                query += " WHERE user_id = %s"
                params.append(session['user_id'])

                cursor.execute(query, tuple(params))
                db.commit()

                if name and name.strip():
                    session['username'] = name.strip()

                flash('Profile updated successfully!', 'success')
            else:
                flash('No changes to update.', 'info')

    except Exception as e:
        logger.error(f"Profile update error: {e}")
        db.rollback()
        flash('An error occurred while updating your profile.', 'error')

    return redirect(url_for('dashboard'))

@app.route('/logout', methods=['POST'])
def logout():
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
        logger.error(f"Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/health/db')
def health_check_db():
    try:
        db = get_db()
        with db.cursor() as cursor:
            cursor.execute('SELECT 1')
        return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        init_db_pool()
    port = int(os.getenv('PORT', 5002))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True 
    )