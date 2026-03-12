RAG usecase with Multiple Ingestion Pipeline and multiple vector stores

run command "pip install -r requirements.txt"

then run these 3 command sto create their seperate vector stores, as multi RAG Agent can have muliple vector stores
"python ingestion/ingest_python.py"
"python ingestion/ingest_node.py"
"python ingestion/ingest_mern.py"

while creating vectors, the vectors will be created in vectorstores folder, (kept mine vector stores for reference, you can delete all the nested folders in vectorstores folder as the vector store files will be created again by running the above command)

then run "python app.py"

ask any questions like 
1) what is nodejs
2) what is multi threading in Java
3) what are closures in JS
4) what is eventEmitter in nodejs?