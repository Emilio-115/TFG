from src.xgboost_impl.main import main as xgb_main
from src.dataset_builder.dataset_builder import main as generate_data
from src.cnn.main import main as cnn_main
from src.lstm.main import main as lstm_main
import tensorflow as tf


def main():
    print("1. Dataset builder agg")
    print("2. Dataset builder")
    print("3. XGBoost")
    print("4. Res Net 1D")
    print("5. TCN")
    print("6. BiLSTM")
    print("7. Entrenar todas las redes")
    print("8. Entrenar todos los modelos")

    option = input("Elige opción: ")

    if option == "1":
        print(f'{"#"*30+"\n"}Construyendo dataset agregado\n{"#"*30}')
        generate_data()
    elif option == "2":
        print(f'{"#"*30}\nConstruyendo datos(no se almacenan)\n{"#"*30}')
        generate_data(agg=False)
    elif option == "3":
        print(f'{"#"*30}\nEjecutando XGBoost\n{"#"*30}')
        xgb_main()
    elif option == "4":
        print(f'{"#"*30}\nEjecutando Res Net 1D\n{"#"*30}')
        cnn_main()
    elif option == "5":
        print(f'{"#"*30}\nEjecutando TCN\n{"#"*30}')
        cnn_main(res_net=False)
    elif option == "6":
        print(f'{"#"*30}\nEjecutando BiLSTM\n{"#"*30}')
        lstm_main()
    elif option == "7":
        print(f'{"#"*30}\nEjecutando todas las redes\n{"#"*30}')
        
        print(f'{"#"*30}\nEjecutando Res Net 1D\n{"#"*30}')
        cnn_main()

        print(f'{"#"*30}\nEjecutando TCN\n{"#"*30}')
        cnn_main(res_net=False)
        
        print(f'{"#"*30}\nEjecutando BiLSTM\n{"#"*30}')
        lstm_main()
    elif option == "8":
        print(f'{"#"*30}\nEjecutando todas las implementaciones\n{"#"*30}')
        
        print(f'{"#"*30}\nEjecutando XGBoost\n{"#"*30}')
        xgb_main()
        
        print(f'{"#"*30}\nEjecutando Res Net 1D\n{"#"*30}')
        cnn_main()

        print(f'{"#"*30}\nEjecutando TCN\n{"#"*30}')
        cnn_main(res_net=False)
        
        print(f'{"#"*30}\nEjecutando BiLSTM\n{"#"*30}')
        lstm_main()
    else:
        print("Opción invalida.")

if __name__ == "__main__":
    main()