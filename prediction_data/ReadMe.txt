'''CASO 1 - si el paciente contagiado de CoV-2 necesitará hospitalización'''
label: TIPO_PACIENTE

'''CASO 2 -  Mortalidad de los contagiagos ANTES de ir al hospital'''
//Se descartaron las personas que si hayan ido al hospital (no se si la precision influya, o solo descartar la columna de hosp para incluir mas datos)
label: BOOL_DEF

'''CASO 3 -  Mortalidad de los contagiagos DESPUES de ir al hospital'''
//Se descartaron las personas que no hayan ido al hospital (no se si la precision influya, o solo incluir la columna de hosp para incluir mas datos)
label: BOOL_DEF

'''CASO 4 -Necesidad de UCI ANTES de diagnostico de neumonia'''     
label: UCI

'''CASO 5 -Necesidad de UCI DESPUES de diagnostico de neumonia'''     
label: UCI

'''CASO 6 - necesidad de ventilador ANTES de DIAGNOSTICO de neumonia e ICU'''     
label: INTUBADO

'''CASO 7 - necesidad de ventilador DESPUES de DIAGNOSTICO de neumonia e ICU'''     
label: INTUBADO
