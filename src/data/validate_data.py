import great_expectations as ge
from typing import Tuple, List


def validate_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("Starting data validation with Great Expectations...")
    
    # convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # validate existence of all required columns
    print("Validating schema and required columns...")
    ge_df.expect_column_to_exist("customer_id")
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("age") 
    ge_df.expect_column_to_exist("subscription_type")
    ge_df.expect_column_to_exist("watch_hours")
    ge_df.expect_column_to_exist("last_login_days")
    ge_df.expect_column_to_exist("region")
    ge_df.expect_column_to_exist("device")
    ge_df.expect_column_to_exist("monthly_fee")
    ge_df.expect_column_to_exist("payment_method")
    ge_df.expect_column_to_exist("number_of_profiles")
    ge_df.expect_column_to_exist("avg_watch_time_per_day")
    ge_df.expect_column_to_exist("favorite_genre")
    ge_df.expect_column_to_exist("churned")
    
    # validate the target column
    ge_df.expect_column_values_to_be_in_set("churned", [0, 1])

    # validate categorical column values
    print("Validating unique column values...") 
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female", "Other"])
    ge_df.expect_column_values_to_be_in_set("subscription_type", ['Basic', 'Standard', 'Premium'])
    ge_df.expect_column_values_to_be_in_set("region", ['Africa', 'Europe', 'Asia', 'Oceania', 'South America', 'North America'])
    ge_df.expect_column_values_to_be_in_set("device", ['TV', 'Mobile', 'Laptop', 'Desktop', 'Tablet'])
    ge_df.expect_column_values_to_be_in_set("payment_method", ['Gift Card', 'Crypto', 'Debit Card', 'PayPal', 'Credit Card'])
    ge_df.expect_column_values_to_be_in_set("favorite_genre", ['Action', 'Sci-Fi', 'Drama', 'Horror', 'Romance', 'Comedy', 'Documentary'])

    # validate numerical column values
    ge_df.expect_column_values_to_be_between("age", min_value=12, max_value=120)
    ge_df.expect_column_values_to_be_between("watch_hours", min_value=0)
    ge_df.expect_column_values_to_be_between("last_login_days", min_value=0)
    ge_df.expect_column_values_to_be_between("monthly_fee", min_value=0, max_value=20)
    ge_df.expect_column_values_to_be_between("number_of_profiles", min_value=1, max_value=5)
    ge_df.expect_column_values_to_be_between("avg_watch_time_per_day", min_value=500)
    
    
    # run the complete validation suite
    print("Running complete validation suite...")
    results = ge_df.validate()
    
    # catch all failed expectations
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations
