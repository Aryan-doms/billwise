-- SQL schema for the database

-- Users Table (renamed from User to avoid reserved keyword)
CREATE TABLE IF NOT EXISTS Users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    username VARCHAR(255) NOT NULL,
    profile_picture BYTEA
);

-- Document Table
CREATE TABLE IF NOT EXISTS Document (
    doc_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    filename VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'Pending',
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- Bill Table
CREATE TABLE IF NOT EXISTS Bill (
    bill_id SERIAL PRIMARY KEY,
    doc_id INTEGER,  -- Changed from NOT NULL to nullable
    user_id INTEGER NOT NULL,
    category VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    invoice_date DATE,
    confidence_level VARCHAR(20), 
    FOREIGN KEY (doc_id) REFERENCES Document(doc_id),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

-- Monthly Budget Table
CREATE TABLE IF NOT EXISTS MonthlyBudget (
    budget_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    month VARCHAR(7) NOT NULL,  -- in 'YYYY-MM' format
    budget DECIMAL(10,2) DEFAULT 0,
    spent DECIMAL(10,2) DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES Users(user_id),
    UNIQUE(user_id, month)
);