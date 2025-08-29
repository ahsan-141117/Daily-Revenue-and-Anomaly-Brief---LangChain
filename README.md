# Please credit **Mohammad Ahsan Hummayoun** and **Atsushi Hashimoto** when using, sharing, or adapting this code 

This project automates the generation of a **Daily Revenue & Anomaly Brief** for e-commerce data.  
It computes KPIs (Revenue, Orders, AOV), compares results against a rolling baseline, and detects anomalies.  
A concise executive-style HTML email is then produced with a professional chart and summary.  

This workflow can be scheduled to operate on different intervals and applied to retail and e-commerce stores worldwide.

# ⚙️ How it Works

**Data Ingestion**  
Reads transactional data (CSV or other sources) containing order IDs, order dates, and totals.  

**KPI Calculation**  
Computes key performance indicators:  
- Total Revenue  
- Number of Orders  
- Average Order Value (AOV)  

**Baseline Comparison & Anomaly Detection**  
Compares daily metrics against a rolling historical baseline to flag anomalies (spikes or drops).  

**LLM Summary Generation**  
Integrates **Groq Cloud** via **LangChain** to generate a concise daily executive brief.  
(Other LLM providers can also be swapped in with LangChain’s unified interface.)   

**Chart Creation**  
Leverages **QuickChart** to produce professional bar/line charts embedded into the email.  

**Email Delivery**  
Sends the daily report as a styled HTML email via **Gmail SMTP**.  
