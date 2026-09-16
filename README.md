# Taller 10: SVM y fronteras no lineales

Aplicación gráfica en Python para desarrollar el laboratorio de SVM.

## Ejecución

```powershell
py -m pip install -r requirements.txt
py app_svm_taller.py
```

## Guía del taller

1. Entrena el conjunto inicial con el kernel lineal y observa los vectores de soporte.
2. Agrega el punto `[5, 5]` como clase `0`.
3. Entrena de nuevo con el kernel lineal: la separación queda forzada.
4. Selecciona RBF, entrena y compara la frontera curva obtenida.
5. Prueba predicciones, por ejemplo el punto `[5, 4]`.

El caso representa situaciones en que una clase queda rodeada por otra. Por ejemplo, al combinar indicadores médicos, un paciente con un patrón de riesgo específico puede quedar dentro de la zona típica de pacientes sanos; una frontera lineal no lo aísla bien y RBF puede modelar ese límite curvo.
