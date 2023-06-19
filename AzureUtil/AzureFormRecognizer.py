import os, json
import html
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class AzureFormRecognizerLayout:

    def __init__(self):
        self.endpoint = os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"]
        self.key = os.environ["AZURE_FORM_RECOGNIZER_KEY"]

        self.document_analysis_client = DocumentAnalysisClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )

    def extract_local_single_file(self, file_name: str):
        not_completed = True
        while not_completed:
            with open(file_name, "rb") as f:
                poller = self.document_analysis_client.begin_analyze_document(
                    "prebuilt-layout", document=f
                )
                not_completed=False
        result = poller.result()
        return result
    
    def extract_content_from_url(self,document_url):
        """Returns the text content of the file at the given URL."""
        #print("Analyzing", document_url)
        
        poller = self.document_analysis_client.begin_analyze_document_from_url(
            "prebuilt-layout", document_url)
        result = poller.result()
        return result

    def extract_files(self, folder_name: str, destination_folder_name: str):
        os.makedirs(destination_folder_name, exist_ok=True)
        for file in os.listdir(folder_name):
            if file[-3:].upper() in ['PDF','JPG','PNG']:
                print('Processing file:', file, end='')
            
                file_name = os.path.join(folder_name, file)
                result = self.extract_local_single_file(file_name)
                page_content = self.get_page_content(file_name, result)
                output_file = os.path.join(destination_folder_name, file[:-3] +'json')
                print(f'  write output to {output_file}')
                with open(output_file, "w") as f:
                    f.write(json.dumps({'filename':file_name, 'content':page_content}, indent=4))

    # This method is more suitable for "read" api
    def get_page_content(self, result):
        page_content = []
        for page in result.pages:

            all_lines_content = []
            for line_idx, line in enumerate(page.lines):
                all_lines_content.append(' '.join([word.content for word in line.get_words()]))
            page_content.append({'page_number':page.page_number, 
                                    'page_content':' '.join(all_lines_content)})

        return page_content
    
    def table_to_html(self, table):
        table_html = "<table>"
        rows = [sorted([cell for cell in table.cells if cell.row_index == i], key=lambda cell: cell.column_index) for i in range(table.row_count)]
        for row_cells in rows:
            table_html += "<tr>"
            for cell in row_cells:
                tag = "th" if (cell.kind == "columnHeader" or cell.kind == "rowHeader") else "td"
                cell_spans = ""
                if cell.column_span > 1: cell_spans += f" colSpan={cell.column_span}"
                if cell.row_span > 1: cell_spans += f" rowSpan={cell.row_span}"
                table_html += f"<{tag}{cell_spans}>{html.escape(cell.content)}</{tag}>"
            table_html +="</tr>"
        table_html += "</table>"
        return table_html
    
    def get_document_text(self, result):
        offset = 0
        page_map = []

        for page_num, page in enumerate(result.pages):
            tables_on_page = [table for table in result.tables if table.bounding_regions[0].page_number == page_num + 1]

            # mark all positions of the table spans in the page
            page_offset = page.spans[0].offset
            page_length = page.spans[0].length
            table_chars = [-1]*page_length
            for table_id, table in enumerate(tables_on_page):
                for span in table.spans:
                    # replace all table spans with "table_id" in table_chars array
                    for i in range(span.length):
                        idx = span.offset - page_offset + i
                        if idx >=0 and idx < page_length:
                            table_chars[idx] = table_id

            # build page text by replacing charcters in table spans with table html
            page_text = ""
            added_tables = set()
            for idx, table_id in enumerate(table_chars):
                if table_id == -1:
                    page_text += result.content[page_offset + idx]
                elif not table_id in added_tables:
                    page_text += self.table_to_html(tables_on_page[table_id])
                    added_tables.add(table_id)

            page_text += " "
            page_map.append((page_num, offset, page_text))
            offset += len(page_text)            
        return page_map

    def _get_azure_container(self, blob_service_client, container_name:str):
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
        except:
            pass
        
    def analyze_read_azure_blob(self, input_folder: str, text_output_folder: str, intemediate_output_folder: str = None):
    
        storage = os.environ["AZURE_BLOB_STORAGE_ACCOUNT_NAME"]
        key = os.environ["AZURE_BLOB_STORAGE_KEY"]
        connect_str = f"DefaultEndpointsProtocol=https;AccountName={storage};AccountKey={key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        # Container client for raw container.
        raw_container_client = blob_service_client.get_container_client(input_folder)

        # Container client for intemediate
        if intemediate_output_folder != None:
            intemediate_container_client = self._get_azure_container(blob_service_client, intemediate_output_folder)

        text_container_client = self._get_azure_container(blob_service_client, text_output_folder)

        storageUrlBase = raw_container_client.primary_endpoint

        blob_list = raw_container_client.list_blobs()
        for blob in blob_list:
            blobUrl = f'{storageUrlBase}/{blob.name}'
            print(blobUrl)
            poller = self.form_recognizer_client.begin_analyze_document_from_url("prebuilt-document", blobUrl)

            # Get results
            doc = poller.result()

            if intemediate_output_folder!= None:
                intemediate_blob_client = intemediate_container_client.get_blob_client(container=intemediate_output_folder, blob=blob.name)
                intemediate_blob_client.upload_blob(doc, blob.blob_type, overwrite=True)
            
            # Create a blob client for the new file
            blob_client = text_container_client.get_blob_client(container=text_output_folder, blob=blob.name)
            blob_client.upload_blob(blob, blob.blob_type, overwrite=True)

            # Delete blob from raw now that its processed
            # raw_container_client.delete_blob(blob)
        # return docs