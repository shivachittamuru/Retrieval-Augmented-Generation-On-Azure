import os, json
from langchain.llms import AzureOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

class Summarization:

    def __init__(self, deployment_name, temperature = 0, max_tokens=256) -> None:
        self.llm = AzureOpenAI(temperature=temperature, deployment_name=deployment_name, max_tokens=max_tokens)
        

    def get_summary(self, file, summarization_type='map_reduce'):

        with open(file) as f:
            file_json = json.loads(f.read())

        docs = [Document(page_content = page['page_content']) for page in file_json['content']]
        chain = load_summarize_chain(self.llm, chain_type=summarization_type)
        return chain.run(docs)
            
    def generate_summary_files(self, source_folder, destination_folder, summarization_type='map_reduce', verbose=True):
        os.makedirs(destination_folder, exist_ok=True)

        all_summaries = []
        for file in os.listdir(source_folder):
            if verbose:
                print('\n\nGenerate summary for file:', file)
            file_summary = self.get_summary(os.path.join(source_folder, file), summarization_type=summarization_type)
            summary = {'file': file, 'summarization_type':summarization_type, 'summary':file_summary}
            with open(os.path.join(destination_folder, file), "w") as f:
                f.write(json.dumps(summary))
            if verbose:
                print(summary)
            all_summaries.append(summary)
        return all_summaries
    
