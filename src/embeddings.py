from src.RAG import VectorEmbeddings, DocumentType

# Crear instancia
legal_db = VectorEmbeddings("normativa_tributaria-RAG")
legal_db.move_to_gpu() 

# Procesar Ley General Tributaria
lgt_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-2003-23186",
    document_type=DocumentType.GENERAL,
    document_id="LGT",
    metadata={
        "título": "Ley 58/2003, de 17 de diciembre, General Tributaria",
        "año": 2003,
        "ámbito": "Nacional",
        "rango": "Ley"
    }
)

# Ley 39/2015, de 1 de octubre, del Procedimiento Administrativo Común de las Administraciones Pública
proc_admin_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-2015-10565",
    document_type=DocumentType.GENERAL,
    document_id="PROC_ADMIN",
    metadata={
        "título": "Ley 39/2015, del Procedimiento Administrativo Común",
        "año": 2015,
        "ámbito": "Nacional",
        "rango": "Ley"
    }
)

# Ley 40/2015, de 1 de octubre, de Régimen Jurídico del Sector Público.
reg_juridico_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-2015-10566",
    document_type=DocumentType.GENERAL,
    document_id="REG_JURIDICO",
    metadata={
        "título": "Ley 40/2015, de Régimen Jurídico del Sector Público",
        "año": 2015,
        "ámbito": "Nacional",
        "rango": "Ley"
    }
)

# Ley 22/2009, de 18 de diciembre, por la que se regula el sistema de financiación de las Comunidades Autónomas de régimen común y Ciudades con Estatuto de Autonomía y se modifican determinadas normas tributarias.
financiacion_ccaa_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-2009-20375",
    document_type=DocumentType.GENERAL,
    document_id="FINANCIACION_CCAA",
    metadata={
        "título": "Ley 22/2009, de financiación de las Comunidades Autónomas",
        "año": 2009,
        "ámbito": "Nacional",
        "rango": "Ley"
    }
)

### ITPAJD
# Real Decreto Legislativo 1/1993, de 24 de septiembre, por el que se aprueba el Texto refundido de la Ley del Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados.
itpajd_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1993-25359",
    document_type=DocumentType.ITPAJD,
    document_id="RD_ITPAJD",
    metadata={
        "título": "Real Decreto Legislativo 1/1993, de la Ley del Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados",
        "año": 1993,
        "ámbito": "Nacional",
        "rango": "Real Decreto Legislativo"
    },
    related_docs=["LGT", "FINANCIACION_CCAA"]
)

# Real Decreto 828/1995, de 29 de mayo, por el que se aprueba el Reglamento del Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados.
rd_828_1995_id = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1995-15071",
    document_type=DocumentType.ITPAJD,
    document_id="RD_828_1995",
    metadata={
        "título": "Real Decreto 828/1995, de 29 de mayo, por el que se aprueba el Reglamento del Impuesto sobre Transmisiones Patrimoniales y Actos Jurídicos Documentados",
        "año": 1995,
        "ámbito": "Nacional",
        "rango": "Real Decreto"
    },
    related_docs=["RD_ITPAJD"]  
)

## ISD
# Ley 29/1987, de 18 de diciembre, del Impuesto sobre Sucesiones y Donaciones
Ley_ISD = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1987-28141",
    document_type=DocumentType.ISD,
    document_id="Ley_29_1987",
    metadata={
        "título": "Ley 29/1987, de 18 de diciembre, del Impuesto sobre Sucesiones y Donaciones",
        "año": 1987,
        "ámbito": "Nacional",
        "rango": "Ley"
    },
    related_docs=["LGT", "FINANCIACION_CCAA"] 
)

# Real Decreto 1629/1991, de 8 de noviembre, por el que se aprueba el Reglamento del Impuesto sobre Sucesiones y Donaciones.
Reglamento_ISD = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1991-27678",
    document_type=DocumentType.ISD,
    document_id="RD_1629_1991",
    metadata={
        "título": "Real Decreto 1629/1991, de 8 de noviembre, por el que se aprueba el Reglamento del Impuesto sobre Sucesiones y Donaciones",
        "año": 1991,
        "ámbito": "Nacional",
        "rango": "Real Decreto"
    },
    related_docs=["Ley_29_1987"]  
)

# Resolución 2/1999, de 23 de marzo, de la Dirección General de Tributos, relativa a la aplicación de las reducciones en la base imponible del Impuesto sobre Sucesiones y Donaciones, en materia de vivienda habitual y empresa familiar.
Resolucion_2_1999 = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1999-8180",
    document_type=DocumentType.ISD,
    document_id="Resolucion_2_1999",
    metadata={
        "título": "Resolución 2/1999, de 23 de marzo, de la Dirección General de Tributos, relativa a la aplicación de las reducciones en la base imponible del Impuesto sobre Sucesiones y Donaciones, en materia de vivienda habitual y empresa familiar",
        "año": 1999,
        "ámbito": "Nacional",
        "rango": "Resolución"
    },
    related_docs=["Ley_29_1987", "RD_1629_1991"] 
)

## Patrimonio
# Ley 19/1991, de 6 de junio, del Impuesto sobre el Patrimonio.
Ley_Patrimonio = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOE-A-1991-14392",
    document_type=DocumentType.IP,
    document_id="Ley_Patrimonio",
    metadata={
        "título": "Ley 19/1991, de 6 de junio, del Impuesto sobre el Patrimonio",
        "año": 1991,
        "ámbito": "Nacional",
        "rango": "Ley"
    },
    related_docs=["LGT", "FINANCIACION_CCAA"] 
)

## Comunidad Autónoma de Cantabria
#Cedidos
cedidos_cantabria = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOCT-c-2008-90028",
    document_type=DocumentType.CANTABRIA,
    document_id="cedidos_cantabria",
    metadata={
        "título": "Decreto Legislativo 62/2008, de 19 de junio, por el que se aprueba el texto refundido de la Ley de Medidas Fiscales en materia de Tributos cedidos por el Estado",
        "año": 2008,
        "ámbito": "Autonómico - Cantabria",
        "rango": "Decreto Legislativo"
    },
    related_docs=["FINANCIACION_CCAA"] 
)


## Comunidad Autónoma de Canarias
#Cedidos
cedidos_canarias = legal_db.process_document(
    source="https://www.boe.es/buscar/act.php?id=BOC-j-2009-90008",
    document_type=DocumentType.CANARIAS,
    document_id="cedidos_canarias",
    metadata={
        "título": "Decreto-Legislativo 1/2009, de 21 de abril, por el que se aprueba el Texto Refundido de las disposiciones legales vigentes dictadas por la Comunidad Autónoma de Canarias en materia de tributos cedidos.",
        "año": 2009,
        "ámbito": "Autonómico - Canarias",
        "rango": "Decreto Legislativo"
    },
    related_docs=["FINANCIACION_CCAA"]
)
