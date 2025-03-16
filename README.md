# BillWise

BillWise is a Flask-based web application designed for efficient bill processing and expense management. It enables users to upload bills in various formats, extracts relevant details using AI-powered OCR and NLP, and provides insightful expense summaries.

## Features
- **Bill Upload & Processing:** Supports images and PDFs.
- **AI-Powered Extraction:** Uses Google Gemini API for OCR and NLP analysis.
- **Expense Categorization:** Automatically categorizes expenses.
- **Budget Tracking:** Allows users to monitor spending and set budgets.
- **User Dashboard:** Displays visual insights and history of processed bills.

## Project Structure

```
BillWise/
├── api/
│   └── index.py              # Entry point for Vercel serverless
├── src/
│   ├── app.py               # Main Flask application
│   ├── schema.sql           # Database schema
│   └── website/
│       ├── static/          # CSS, JS, images
│       └── templates/       # HTML files
├── .env                     # Environment variables (don't commit)
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
├── requirements.txt         # Project dependencies
└── vercel.json              # Vercel configuration
```

## Installation

1. **Clone the Repository:**
   ```sh
   git clone <repository-url>
   cd BillWise
   ```

2. **Create a Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Create a `.env` file and add necessary configurations like API keys.

## Usage

1. **Run the Application:**
   ```sh
   python src/app.py
   ```

2. **Access the Web App:**
   Open `http://127.0.0.1:5000` in your browser.

3. Follow the setup prompts and deploy the project.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

