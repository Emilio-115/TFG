from src.xgboost_impl.main import main as xgb_main
def main():
    print("1. XGBoost")
    print("2. Dataset builder")

    option = input("Elige opción: ")

    if option == "1":
        
        xgb_main()
    elif option == "2":
        # from src.dataset_builder.dataset_builder import main as db_main
        # db_main()
        print("A")

if __name__ == "__main__":
    main()