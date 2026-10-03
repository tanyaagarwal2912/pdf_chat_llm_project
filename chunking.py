text = """
Our Data Analyst course lasts for 6 months.
Students get access to all recorded classes.
Live classes happen from Monday to Friday.
Refund requests are accepted within 7 days of purchase.
Students must complete all assignments before placement support starts
Eligible students receive minimum 3 interview opportunities.
"""
chunks = text.strip().split("\n")
for chunk in chunks:
    print("CHUNK:")
    print(chunk)
    print()