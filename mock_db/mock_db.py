import random

class MockDB:
    def __init__(self):
        pass
    
    def query_plate(self, plate_number)-> dict:
        
        result = {
            "Found": False,
            "Owner": {},
            "Violation": ""
        }
        if random.random() < 0.5:
            result["Found"] = True
            result["Owner"] = {
                "Name": "John Doe",
                "Address": random.choice(["Shivbari road ", "NH 40 Chargola", "789 Silchar","123 Main St"]),
                "Phone": "555-1234"
            }
            result["Violation"] = random.choice(["Speeding", "Red Light Violation", "Illegal Parking"])
        return result