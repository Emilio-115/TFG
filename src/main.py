from src.xgboost_impl.main import main as xgb_main
from src.dataset_builder.dataset_builder import main as df_agg_main

def main():
    print("1. Dataset builder agg")
    print("2. Dataset builder")
    print("3. XGBoost")
    print("4. CNN")
    print("5. LSTM")


    option = input("Elige opción: ")

    if option == "1":
        print(f'{"#"*30+"\n"}Construyendo dataset agregado\n{"#"*30}')
        df_agg_main()
    elif option == "2":
        print(f'{"#"*30}\nConstruyendo dataset\n{"#"*30}')
    elif option == "3":
        print(f'{"#"*30}\nEjecutando XGBoost\n{"#"*30}')
        xgb_main()
    elif option == "4":
        print(f'{"#"*30}\nEjecutando CNN\n{"#"*30}')
    elif option == "5":
        print(f'{"#"*30}\nEjecutando LSTM\n{"#"*30}')

if __name__ == "__main__":
    main()