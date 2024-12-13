import streamlit as st
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
from crewai_tools import YoutubeChannelSearchTool, ScrapeWebsiteTool
import warnings
from typing import List


warnings.filterwarnings("ignore")
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq model
groq_model = ChatGroq(model="groq/llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# Set up Streamlit page
st.set_page_config(page_title="🚀 Personalized Stock Insights Generator", layout="wide")
st.title("🚀 Personalized Stock Insights Generator")
st.markdown("""
    Dive deep into personalized stock market insights. 
    Enter a company, stock symbol, or investment topic to get a comprehensive analysis.
""")

# Input area for stock topic
stock_topic = st.text_input(
    "Enter Company Name, Stock Symbol, or Investment Topic",
    placeholder="e.g., Apple, AAPL, AI Stocks, Tech Sector",
    help="Provide a specific stock, company, or investment theme for targeted insights"
)

def create_dynamic_tools(topic: str) -> List:
    """Create tools tailored to the user's stock topic"""
    try:
        youtube_tool = YoutubeChannelSearchTool(
            youtube_channel_handle='@YahooFinance',
            config={
                'llm': {
                    'provider': 'groq',
                    'config': {
                        'model': 'llama-3.1-8b-instant',
                        'api_key': GROQ_API_KEY,
                        'temperature': 0.7
                    }
                },
                'embedder': {
                    'provider': 'google',
                    'config': {
                        'model': 'models/embedding-001'
                    }
                }
            }
        )
        
        web_search_tool = ScrapeWebsiteTool(
            config={
                'llm': {
                    'provider': 'groq',
                    'config': {
                        'model': 'llama-3.1-8b-instant',
                        'api_key': GROQ_API_KEY,
                        'temperature': 0.7
                    }
                }
            }
        )
        
        return [youtube_tool, web_search_tool]
    except Exception as e:
        st.error(f"Error creating tools: {e}")
        return []

def generate_stock_insights(topic):
    """Generate comprehensive stock insights based on user input"""
    # Dynamic tools
    tools = create_dynamic_tools(topic)
    
    # Research Agent
    researcher_agent = Agent(
        role="Stock Market Intelligence Analyst",
        goal=f"Conduct an in-depth investigation and analysis of {topic} in the stock market",
        backstory=f"You are a seasoned financial analyst specializing in researching and providing comprehensive insights about {topic}. "
                  "Your expertise lies in gathering the most recent and relevant information from multiple sources.",
        tools=tools,
        allow_delegation=True,
        llm=groq_model,
        verbose=True
    )
    
    # Insight Compiler Agent
    insights_agent = Agent(
        role="Financial Content Strategist",
        goal="Transform raw financial data into a coherent, engaging narrative",
        backstory="You excel at converting complex financial information into clear, actionable insights. "
                  "Your writing style is professional yet accessible, making complex topics easy to understand.",
        llm=groq_model,
        allow_delegation=False,
        verbose=True
    )
    
    # Report Formatter Agent
    formatter_agent = Agent(
    role="Professional Report Designer",
    goal="Create a visually structured and professionally formatted financial report with tables if necessary but without placeholders, [] page breaks, and images",
    backstory="You specialize in presenting financial information in a clean, organized manner. "
              "Ensure no placeholder text or unnecesary page breaks or [] remains in the final report.",
    tools=[],
    llm=groq_model,
    allow_delegation=False,
    verbose=True,
    config={
        "remove_placeholders": True,
        "strict_formatting": True
    }
)
    
    # Research Task
    research_task = Task(
        description=f"""
        Conduct comprehensive research on {topic}:
        - Latest news and developments
        - Market sentiment
        - Recent performance trends
        - Key financial indicators
        - Potential future outlook
        
        Ensure the information is current, accurate, and provides meaningful insights.
        """,
        expected_output="Detailed raw research findings about the specified stock or topic",
        agent=researcher_agent,
        output_format="Markdown"
    )
    
    # Insight Compilation Task
    insights_task = Task(
        description="Transform the raw research into a compelling narrative. "
                    "Create a story that provides context, analysis, and actionable insights.",
        expected_output="A well-structured, insightful financial narrative",
        agent=insights_agent,
        context=[research_task],
        output_format="Markdown"
    )
    
    # Formatting Task
    format_task = Task(
        description="Refine and format the financial narrative into a professional report. "
                    "Ensure clarity, readability, and professional presentation. No placeholders or unnecessary images , brackets or page breaks should be present",
        expected_output="A polished, professionally formatted financial insights report",
        agent=formatter_agent,
        context=[insights_task],
        output_format="Markdown"
    )
    
    # Create Crew
    crew = Crew(
        agents=[researcher_agent, insights_agent, formatter_agent],
        tasks=[research_task, insights_task, format_task],
        process=Process.sequential,
        verbose=True
    )
    
    # Generate Insights
    result = crew.kickoff(inputs={"topic": topic})
    return result

# Generate Button
if st.button("🔍 Generate Stock Insights"):
    if stock_topic:
        with st.spinner("Generating Comprehensive Stock Insights..."):
            result = generate_stock_insights(stock_topic)
        
        st.success("✅ Insights Generated Successfully!")
        st.markdown("## 📈 Personalized Stock Insights")
        st.markdown(result)
        
        # Download Option
        st.download_button(
            label="📥 Download Stock Insights Report",
            data=str(result),
            file_name=f"{stock_topic.replace(' ', '_')}_stock_insights.md",
            mime="text/plain"
        )
    else:
        st.error("Please enter a stock topic or company name to generate insights.")

# llama-3.1-8b-instant	completion(model="groq/llama-3.1-8b-instant", messages)
# llama-3.1-70b-versatile	completion(model="groq/llama-3.1-70b-versatile", messages)
# llama3-8b-8192	completion(model="groq/llama3-8b-8192", messages)
# llama3-70b-8192	completion(model="groq/llama3-70b-8192", messages)
# llama2-70b-4096	completion(model="groq/llama2-70b-4096", messages)
# mixtral-8x7b-32768	completion(model="groq/mixtral-8x7b-32768", messages)
# gemma-7b-it	completion(model="groq/gemma-7b-it", messages)