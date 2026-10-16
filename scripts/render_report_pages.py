"""Use the packaged render_docx rasterizer with an already exported Word PDF.

Native Word replaces only the unavailable LibreOffice conversion step on Windows.
"""
import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path

from pypdf import PdfReader

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--docx',type=Path,required=True);p.add_argument('--pdf',type=Path,required=True)
p.add_argument('--renderer',type=Path,required=True);p.add_argument('--poppler',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
spec=importlib.util.spec_from_file_location('document_renderer',args.renderer)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def from_word(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    target=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(args.pdf,target)
    return str(target),'PDF exported from the same DOCX by native Microsoft Word after field update'
renderer.convert_to_pdf=from_word
os.environ['PATH']=str(args.poppler)+os.pathsep+os.environ['PATH']
paths=renderer.rasterize(str(args.docx),str(args.output),120,False,False)
pdf=PdfReader(args.pdf)
summary=[]
for i,page in enumerate(pdf.pages,1):
    text=page.extract_text()
    summary.append({'page':i,'characters':len(text),'first_lines':text.splitlines()[:3],'last_lines':text.splitlines()[-3:]})
(args.output/'page_inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Rendered {len(paths)} pages')
print(json.dumps(summary,ensure_ascii=False,indent=2))
