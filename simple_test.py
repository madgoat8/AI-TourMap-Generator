from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "服务正常运行", "status": "ok"}

@app.get("/test")
async def test():
    return {"message": "测试接口正常", "timestamp": "2025-08-25"}

if __name__ == "__main__":
    print("启动简单测试服务...")
    uvicorn.run(app, host="127.0.0.1", port=8002)