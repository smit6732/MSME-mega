"""
scripts/generate_large_sample.py

Dev tool: generates a large synthetic MSME directory CSV, for
reproducing performance measurements against core/duplication.py (see
scripts/profile_pipeline.py). Not shipped as sample data -- regenerate
on demand instead of committing a multi-hundred-KB CSV to the repo.

Usage:
    python scripts/generate_large_sample.py [row_count] [output_path]
    python scripts/generate_large_sample.py 3000 sample_data/large_msme_directory.csv
"""
import random
import csv
import sys

random.seed(7)

ROW_COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "sample_data/large_msme_directory.csv"

BUSINESS_PREFIXES = ["Shree", "Shri", "Om", "Jay", "New", "Modern", "National", "Gujarat",
                      "Krishna", "Ganesh", "Laxmi", "Bharat", "Royal", "Sona", "Star", "United",
                      "City", "Prime", "Classic", "Metro", "Global", "Apex", "Silver", "Golden"]
BUSINESS_CORES = ["Traders", "Textiles", "Enterprises", "Industries", "Hardware", "Foods",
                   "Auto Parts", "Garments", "Plastics", "Steel Works", "Spices", "Exports",
                   "Electronics", "Chemicals", "Engineering", "Pharma", "Agro", "Packaging",
                   "Dairy Farm", "Fashion Hub", "Stationers", "Furniture", "Ceramics", "Timber Mart"]

FIRST_NAMES = ["Ramesh", "Suresh", "Kavita", "Dinesh", "Meena", "Vijay", "Anita", "Manoj",
               "Ashok", "Rina", "Priya", "Nitin", "Farida", "Kiran", "Rakesh", "Sunita"]
LAST_NAMES = ["Patel", "Shah", "Trivedi", "Rao", "Mehta", "Desai", "Nair", "Joshi", "Sheikh", "Kumar"]
CITIES = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Junagadh", "Gandhinagar", "Anand"]
CATEGORIES = ["Manufacturing", "Trading", "Services", "Retail", "Wholesale"]


def random_gst():
    state_code = random.choice(["24", "27", "06", "29"])
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
    digits = "".join(random.choices("0123456789", k=4))
    return f"{state_code}{letters}{digits}A1Z{random.randint(1,9)}"


rows = []
for i in range(ROW_COUNT):
    name = f"{random.choice(BUSINESS_PREFIXES)} {random.choice(BUSINESS_CORES)}"
    rows.append({
        "Business_Name": name,
        "Owner_Name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "GST_Number": random_gst(),
        "Business_Category": random.choice(CATEGORIES),
        "City": random.choice(CITIES),
        "Annual_Turnover": str(random.randint(150000, 9500000)),
        "Registration_Date": f"{random.randint(2015,2023)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "Employee_Count": str(random.randint(2, 60)),
        "Phone_Number": "9" + "".join(random.choices("0123456789", k=9)),
        "Email": name.lower().replace(" ", "") + "@gmail.com",
    })

fieldnames = list(rows[0].keys())
with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> {OUT_PATH}")
