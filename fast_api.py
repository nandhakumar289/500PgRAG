from fastapi import FastAPI
import uvicorn
from rag import orchestrator

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

@app.get("/get_response/{user_query}")
async def read_item(user_query: str = None):
    llm_response = orchestrator(user_query)
    return {"response": llm_response}  
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)