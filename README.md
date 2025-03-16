# My Flask App

This is a Flask application designed to handle bill processing and management. It allows users to upload bills in various formats, processes them, and provides insights into their expenses.

## Project Structure

```
my-flask-app/
├── api/
│   └── index.py              # Entry point for Vercel serverless
├── src/
│   ├── app.py               # Main Flask application
│   ├── schema.sql           # Database schema
│   └── website/
│       ├── static/          # CSS, JS, images
│       └── templates/       # HTML files
├── .env                     # Environment variables (don't commit)
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── vercel.json            # Vercel configuration

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd my-flask-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/app.py
   ```

2. Access the application in your web browser at `http://127.0.0.1:5000`.

## Deployment

This application can be deployed on Vercel. Ensure that the `vercel.json` file is properly configured with the necessary settings.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

