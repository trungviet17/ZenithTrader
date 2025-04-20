from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from low_level_reflection import low_level_reflection
from high_level_reflection import high_level_reflection

# Initialize LLM
llm = OllamaLLM(model="cogito:3b")

# Define Agents
low_level_agent = Agent(
    role="Low-level Reflection Agent",
    goal="Phân tích dữ liệu K-line và thông tin thị trường để đưa ra nhận xét về xu hướng giá.",
    backstory="Bạn là một chuyên gia phân tích kỹ thuật, tập trung vào việc đánh giá dữ liệu thị trường và chỉ báo kỹ thuật để dự đoán xu hướng giá.",
    llm=llm,
    verbose=True
)

high_level_agent = Agent(
    role="High-level Reflection Agent",
    goal="Phân tích các quyết định giao dịch trước đây, đánh giá tính đúng đắn và rút ra bài học để cải thiện.",
    backstory="Bạn là một nhà quản lý danh mục đầu tư, chuyên phân tích hiệu suất giao dịch và đề xuất cải tiến chiến lược.",
    llm=llm,
    verbose=True
)

# Define Tasks
low_level_task = Task(
    description="Phân tích dữ liệu K-line và thông tin thị trường cho mã {symbol}. Trả về nhận xét về xu hướng giá ngắn hạn, trung hạn và dài hạn.",
    agent=low_level_agent,
    expected_output="Một báo cáo văn bản chứa nhận xét về xu hướng giá và các chỉ báo kỹ thuật.",
    execute=lambda inputs: low_level_reflection(inputs["symbol"])
)

high_level_task = Task(
    description="Phân tích danh sách các giao dịch trước đây để đánh giá hiệu suất, rút ra bài học và đề xuất cải tiến.",
    agent=high_level_agent,
    expected_output="Một báo cáo văn bản chứa phân tích hiệu suất, bài học và đề xuất cải tiến.",
    execute=lambda inputs: high_level_reflection(inputs["trades"])
)

# Define Crew
crew = Crew(
    agents=[low_level_agent, high_level_agent],
    tasks=[low_level_task, high_level_task],
    verbose=True
)

# Execute Crew
if __name__ == "__main__":
    inputs = {
        "symbol": "AAPL",
        "trades": [
            {"timestamp": "2023-10-01", "symbol": "AAPL", "decision": "Buy", "price": 150.0},
            {"timestamp": "2023-10-02", "symbol": "AAPL", "decision": "Sell", "price": 155.0},
            {"timestamp": "2023-10-03", "symbol": "MSFT", "decision": "Buy", "price": 300.0},
            {"timestamp": "2023-10-04", "symbol": "MSFT", "decision": "Sell", "price": 290.0}
        ]
    }
    result = crew.kickoff(inputs=inputs)
    print("Kết quả từ CrewAI:")
    for task_output in result:
        print(task_output)