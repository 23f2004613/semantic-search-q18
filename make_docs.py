import json
import PyPDF2

docs = []
for i in range(57):  # Your files
    with open(f"api_doc_{i}.pdf", "rb") as f:  # Or .md
        pdf = PyPDF2.PdfReader(f)
        content = " ".join(page.extract_text() for page in pdf.pages)
        docs.append({"id": i, "content": content[:5000], "metadata": {"source": f"doc_{i}"}})

with open("docs.json", "w") as f:
    json.dump(docs, f)
print("✅ 57 docs ready!")
