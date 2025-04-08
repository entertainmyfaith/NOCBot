import sqlite3
import random
import re

# Configuration
DATABASE_CONFIG = {
    'db_name': 'customers_db.sqlite',
    'table_name': 'Users'
}

# Name lists
usa_names = [
    "Liam Johnson", "Emma Smith", "Noah Brown", "Olivia Jones", "William Garcia",
    "Ava Miller", "James Davis", "Isabella Wilson", "Benjamin Rodriguez", "Sophia Martinez",
    "Michael Anderson", "Mia Taylor", "Elijah Thomas", "Charlotte Moore", "Daniel Jackson",
    "Amelia White", "Matthew Harris", "Evelyn Martin", "Joseph Thompson", "Abigail Perez"
]

uk_names = [
    "Oliver Smith", "Amelia Jones", "Harry Williams", "Isla Brown", "Jack Taylor",
    "Emily Wilson", "George Davies", "Poppy Robinson", "Noah Wright", "Ella Green",
    "Jacob Hall", "Sophie Adams", "Charlie Hill", "Lily Baker", "Thomas Nelson",
    "Grace Carter", "Oscar Mitchell", "Freya Roberts", "James Clark", "Chloe Turner"
]

nepal_names = [
    "Prakash Gurung", "Sita Sharma", "Ramesh Thapa", "Rita Karki", "Suresh Adhikari",
    "Sunita Bhattarai", "Bikash Rai", "Anjali Khadka", "Roshan Basnet", "Puja Subedi",
    "Sabin Tamang", "Sarita Raut", "Dipesh Yadav", "Kabita Giri", "Bibek Pandey",
    "Laxmi Chauhan", "Himal Bhandari", "Sangita Khanal", "Milan Acharya", "Deepa Poudel"
]

india_names = [
    "Aryan Kumar", "Aanya Sharma", "Vivaan Singh", "Diya Verma", "Arjun Patel",
    "Siya Joshi", "Reyansh Reddy", "Aaradhya Gupta", "Ishaan Mehta", "Kiara Iyer",
    "Rohan Pillai", "Shanaya Kapoor", "Neil Desai", "Myra Chatterjee", "Advik Khanna",
    "Anika Bajaj", "Kabir Malhotra", "Navya Srinivasan", "Veer Kapoor", "Anvi Rao"
]

canada_names = [
    "Owen MacDonald", "Chloe Tremblay", "Ethan Gagnon", "Sophie Roy", "Nathan Cote",
    "Emma Lavoie", "William Gauthier", "Olivia Belanger", "Samuel Leblanc", "Alice Moreau",
    "Jackson Fortin", "Florence Gagné", "Lucas Pelletier", "Rosalie Boucher", "Logan Leclerc",
    "Jade Paquette", "Thomas Simard", "Charlie Ouellet", "Noah Bergeron", "Léa Morin"
]


def create_database():
    conn = sqlite3.connect(DATABASE_CONFIG['db_name'])
    cursor = conn.cursor()

    # Create table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE_CONFIG['table_name']} (
            account_number TEXT PRIMARY KEY,
            customer_name TEXT,
            account_type TEXT
        )
    """)

    # Generate and insert 100 users
    all_names = (
        random.sample(usa_names, 20) +
        random.sample(uk_names, 20) +
        random.sample(nepal_names, 20) +
        random.sample(india_names, 20) +
        random.sample(canada_names, 20)
    )
    random.shuffle(all_names)

    account_types = ["residential", "business"]  # Only residential or business
    used_account_numbers = set()

    for i in range(100):
        while True:
            account_number = str(random.randint(1000, 9999))  # Generate random 4-digit number
            if account_number not in used_account_numbers:
                used_account_numbers.add(account_number)
                break  # Exit the loop if the account number is unique

        customer_name = all_names[i]
        account_type = random.choice(account_types)

        cursor.execute(f"""
            INSERT INTO {DATABASE_CONFIG['table_name']} (account_number, customer_name, account_type)
            VALUES (?, ?, ?)
        """, (account_number, customer_name, account_type))

    conn.commit()
    conn.close()
    print("Database created successfully with 100 users.")


def determine_customer_type(transcription):
    """
    Determines the customer type from a transcription by extracting the account number.
    This function is similar to the one in the original script, adapted to work standalone.
    """
    conn = sqlite3.connect(DATABASE_CONFIG['db_name'])
    cursor = conn.cursor()

    if not transcription:
        return "Sorry, I couldn't understand your input."

    account_match = re.search(r'\b(\d{4})\b', transcription)

    if account_match:
        account_number = account_match.group(1)
        print(f"Searching for account: {account_number}")  # Debug output

        cursor.execute(
            f"SELECT customer_name, account_type FROM {DATABASE_CONFIG['table_name']} "
            f"WHERE account_number = ?",
            (account_number,)
        )
        customer_info = cursor.fetchone()
        conn.close()

        if customer_info:
            name, acc_type = customer_info
            return f"Account {account_number} ({name}) is a {acc_type} customer."
        return f"Account {account_number} not found in our system."

    conn.close()
    return "Please provide your 4-digit account number."


if __name__ == "__main__":
    create_database()

    # Example usage of determine_customer_type:
    transcription_example = "My account number is 1234."  # Example with a new random number
    customer_type = determine_customer_type(transcription_example)
    print(customer_type)

