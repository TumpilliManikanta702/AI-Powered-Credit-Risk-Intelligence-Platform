import os
from langchain_groq import ChatGroq
from src.talk_to_data.prompt_templates import SQL_PROMPT, ANSWER_PROMPT
from src.talk_to_data.query_runner import run_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

class NLtoSQLChatbot:
    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is missing.")
            
        # Using temperature 0 for strict generation
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=key,
            model_name="llama-3.3-70b-versatile"
        )
        
    def get_sql_query(self, question: str) -> str:
        prompt = SQL_PROMPT.format(question=question)
        response = self.llm.invoke(prompt)
        # Clean response
        sql_query = response.content.strip()
        # Remove markdown ticks if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
            
        return sql_query.strip()
        
    def generate_business_answer(self, question: str, query: str, result: str) -> str:
        prompt = ANSWER_PROMPT.format(question=question, query=query, result=result)
        response = self.llm.invoke(prompt)
        return response.content.strip()
        
    def process_question(self, question: str) -> dict:
        try:
            logger.info(f"Processing question: {question}")
            sql_query = self.get_sql_query(question)
            logger.info(f"Generated SQL: {sql_query}")
            
            # Execute
            df = run_query(sql_query)
            result_str = df.to_string()
            
            # Generate Answer
            answer = self.generate_business_answer(question, sql_query, result_str)
            
            return {
                "sql_query": sql_query,
                "result_df": df,
                "business_answer": answer,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "sql_query": None,
                "result_df": None,
                "business_answer": f"I encountered an error: {str(e)}",
                "status": "error"
            }
