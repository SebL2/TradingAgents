from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import functools
import time
import json


def create_trader(llm, memory,toolkit):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)
        
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        
        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."
        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_YFin_data_online,
                toolkit.get_stockstats_indicators_report_online,
            ]
        else:
            tools = [
                toolkit.get_YFin_data,
                toolkit.get_stockstats_indicators_report,
            ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}."
             "This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment." 
             "Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
             "Leverage these insights to make an informed and strategic decision."
             "Furthermore, from the given tools: {tool_names}, create a report for {ticker}"
             "using the momentum strategy starting from {current_date}, going back a month. Include the date range in the report"
             "Calculate and analyze the momentum by taking the price difference from back a month, and determine whether if the momentum is positive or negative. Include the RSI as well"
             "Include these data points in the report as well, clearly labelling them in the report, and ensure the next role uses these data points in their analysis."),
            ("user", "You are a trading agent analyzing market data to make investment decisions."
             "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
             "End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation."
             "Do not forget to utilize lessons from past decisions to learn from your mistakes. Here is some reflections from similar situations you traded in and the lessons learned: {past_memory_str}"),
            MessagesPlaceholder(variable_name="messages"),
        ])
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(company_name = company_name)
        prompt = prompt.partial(investment_plan=investment_plan)
        prompt = prompt.partial(past_memory_str= past_memory_str)
        chain = prompt | llm.bind_tools(tools)
        
        result = chain.invoke(state["messages"])
        report = ""

        report = result.content if result.content else "Processing momentum analysis..."

        return {
            "messages": [result],
            "trader_investment_plan": report,
            "sender": name,
        }
    return functools.partial(trader_node, name="Trader")