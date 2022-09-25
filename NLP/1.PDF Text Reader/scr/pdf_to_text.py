# -*- coding: utf-8 -*-

#!pip install pdfplumber -q

import pdfplumber as pp

def get_text_from_pdf(pdf_file_name):
  with pp.open(pdf_file_name) as pdf_file:
    for page_no, page in enumerate(pdf_file.pages, start=1):
      print("page number : " + str(page_no))
      data = page.extract_text()
      print(data.strip())
      print('_'*45)

get_text_from_pdf('/data/journal_1.pdf')

