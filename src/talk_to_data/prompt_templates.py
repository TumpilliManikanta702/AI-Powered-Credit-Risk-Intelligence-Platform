from langchain.prompts import PromptTemplate

SQL_GENERATION_TEMPLATE = """
You are an expert data analyst and SQLite SQL developer.
Generate ONLY a valid SQLite SELECT query based on the user's question.

CRITICAL RULES:
1. NEVER generate DELETE, UPDATE, INSERT, DROP, ALTER, or CREATE queries.
2. Only use the SELECT statement.
3. Your response must contain ONLY the SQL query, without markdown formatting, without backticks, without explanations.
4. If you cannot answer the question using the schema, return "SELECT 'Invalid Query' AS Result;"
5. Use only the columns provided in the schema.

Schema of table `credit_risk_data`:
- TARGET (INTEGER): 1 = Default, 0 = Repaid
- AMT_INCOME_TOTAL (FLOAT): Client's income
- AMT_CREDIT (FLOAT): Credit amount of the loan
- AMT_ANNUITY (FLOAT): Loan annuity
- DAYS_BIRTH (INTEGER): Client's age in days at the time of application (negative)
- DAYS_EMPLOYED (INTEGER): How many days before the application the person started current employment (negative)
- CODE_GENDER (TEXT): Gender of the client
- FLAG_OWN_CAR (TEXT): Flag if the client owns a car
- FLAG_OWN_REALTY (TEXT): Flag if client owns a house or flat
- NAME_INCOME_TYPE (TEXT): Clients income type (Businessman, Working, Maternity leave,...)
- NAME_EDUCATION_TYPE (TEXT): Level of highest education the client achieved
- OCCUPATION_TYPE (TEXT): What kind of occupation does the client have
- CNT_FAM_MEMBERS (FLOAT): How many family members does client have

User Question: {question}

SQL Query:
"""

SQL_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=SQL_GENERATION_TEMPLATE
)

ANSWER_GENERATION_TEMPLATE = """
You are a helpful business data analyst.
You are given a user's question, the SQL query used to find the answer, and the raw result from the database.
Provide a clear, business-readable answer to the user's question based on the result.
Do not mention the SQL query in your response, just answer the question in a friendly manner.

User Question: {question}
SQL Query: {query}
SQL Result: {result}

Business Answer:
"""

ANSWER_PROMPT = PromptTemplate(
    input_variables=["question", "query", "result"],
    template=ANSWER_GENERATION_TEMPLATE
)
