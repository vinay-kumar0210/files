import json
from collections import Counter
from datetime import datetime
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_age(birth_date):
    if not birth_date:
        return None
    birth_date = datetime.strptime(birth_date, "%Y-%m-%dT%H:%M:%S.%fZ")
    today = datetime.now()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def get_age_group(age):
    if age is None:
        return None
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 59:
        return "Adult"
    else:
        return "Senior"

def is_valid_mobile(phone_number):
    phone_number = phone_number.strip().replace(" ", "")
    if phone_number.startswith("+91"):
        phone_number = phone_number[3:]
    elif phone_number.startswith("91"):
        phone_number = phone_number[2:]

    return phone_number.isdigit() and len(phone_number) == 10 and 6000000000 <= int(phone_number) <= 9999999999

def calculate_missing_percentages(data, fields):
    total_records = len(data)
    missing_counts = {field: 0 for field in fields}
    for record in data:
        patient = record.get("patientDetails", {})
        for field in fields:
            if not patient.get(field):
                missing_counts[field] += 1

    return {field: round((count / total_records) * 100, 2) for field, count in missing_counts.items()}

def calculate_female_percentage(data):
    total_records = len(data)
    gender_counts = Counter(record.get("patientDetails", {}).get("gender", "") for record in data)
    mode_gender = gender_counts.most_common(1)[0][0]

    female_count = sum(1 for record in data if record.get("patientDetails", {}).get("gender", mode_gender) == "F")
    return round((female_count / total_records) * 100, 2)

def count_age_groups(data):
    age_groups = {"Child": 0, "Teen": 0, "Adult": 0, "Senior": 0}
    for record in data:
        birth_date = record.get("patientDetails", {}).get("birthDate")
        age = calculate_age(birth_date)
        group = get_age_group(age)
        if group:
            age_groups[group] += 1
    return age_groups

def calculate_average_medicines(data):
    total_medicines = sum(len(record.get("consultationData", {}).get("medicines", [])) for record in data)
    return round(total_medicines / len(data), 2)

def get_third_most_frequent_medicine(data):
    medicine_counter = Counter(
        med["medicineName"]
        for record in data
        for med in record.get("consultationData", {}).get("medicines", [])
    )
    return medicine_counter.most_common(3)[-1][0]

def calculate_medicine_distribution(data):
    active_count = inactive_count = 0
    for record in data:
        for med in record.get("consultationData", {}).get("medicines", []):
            if med.get("isActive"):
                active_count += 1
            else:
                inactive_count += 1

    total = active_count + inactive_count
    if total == 0:
        return (0, 0)
    return (round((active_count / total) * 100, 2), round((inactive_count / total) * 100, 2))

def calculate_valid_mobile_count(data):
    return sum(1 for record in data if is_valid_mobile(record.get("phoneNumber", "")))

def calculate_correlation(data):
    ages, medicine_counts = [], []
    for record in data:
        birth_date = record.get("patientDetails", {}).get("birthDate")
        age = calculate_age(birth_date)
        medicines = record.get("consultationData", {}).get("medicines", [])

        if age is not None:
            ages.append(age)
            medicine_counts.append(len(medicines))

    if len(ages) < 2 or len(medicine_counts) < 2:
        return 0

    return round(np.corrcoef(ages, medicine_counts)[0, 1], 2)

def main():
    data = load_data('DataEngineeringQ2.json')

    missing_percentages = calculate_missing_percentages(data, ["firstName", "lastName", "birthDate"])
    female_percentage = calculate_female_percentage(data)
    age_groups = count_age_groups(data)
    average_medicines = calculate_average_medicines(data)
    third_most_frequent_medicine = get_third_most_frequent_medicine(data)
    medicine_distribution = calculate_medicine_distribution(data)
    valid_mobile_count = calculate_valid_mobile_count(data)
    correlation = calculate_correlation(data)

    results = {
        "missing_percentages": missing_percentages,
        "female_percentage": female_percentage,
        "adult_count": age_groups["Adult"],
        "average_medicines": average_medicines,
        "third_most_frequent_medicine": third_most_frequent_medicine,
        "medicine_distribution": medicine_distribution,
        "valid_mobile_count": valid_mobile_count,
        "correlation": correlation
    }

    print(results)

if __name__ == "__main__":
    main()
