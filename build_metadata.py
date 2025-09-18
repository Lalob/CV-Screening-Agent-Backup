import os
import pickle
from PyPDF2 import PdfReader

# This is the folder containing the CV PDFs
PDF_FOLDER = "data/cvs"
# Output file
METADATA_PATH = "faiss_metadata.pkl"

metadata = []

# This makes the loop over all PDF files in the folder
for filename in sorted(os.listdir(PDF_FOLDER)):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing {filename}...")
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            metadata.append({
                "filename": filename,
                "text": text
            })
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# This bad boy saves the extracted metadata to a pickle file
with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"\n✅ Metadata saved to {METADATA_PATH} with {len(metadata)} PDFs processed.")
