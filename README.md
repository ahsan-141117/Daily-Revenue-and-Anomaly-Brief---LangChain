# Please credit **Mohammad Ahsan Hummayoun** when using, sharing, or adapting this code.  

This project automates the generation of a **Daily Revenue & Anomaly Brief** for e-commerce data.  
It computes KPIs (Revenue, Orders, AOV), compares against a rolling baseline, and detects anomalies.  
A concise executive-style HTML email is then produced with a professional chart and summary.  

This workflow can be scheduled to operate on different intervals. It can be applied to Retail and E-Commerce stores 
worldwide.

To run, set up a `.env` file with the correct variable names and values. You’ll need external APIs:  
- **Groq Cloud** (`GROQ_API_KEY`) for the LLM summary  
- **QuickChart** (no key needed) for rendering the chart  
- **Gmail SMTP** (`GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`) for sending the email

A sample testing dataset (CSV) was used to run the workflow. It has also been attached.
