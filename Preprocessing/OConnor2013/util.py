import re,sys,json

def nicepath(pathstr, html=True):
    d = json.loads(pathstr)
    s = nicepath2(d)
    if not html:
        s = re.sub(r'\<.*?\>', "", s)
        s = re.sub(" +", " ", s)
        s = s.replace("&larr;", "<-").replace("&rarr;", "->")
        s = s.strip()
    return s

def nicepath2(path_arr):
    out = []
    for x in path_arr:
        if x[0]=='A':
            _,rel,arrow = x
            if rel.startswith("prep_"):
                out.append(rel.replace("prep_",""))
            elif rel=='dobj' or rel=='semagent':
                pass
            else:
                deparc = ("&larr;"+rel) if arrow=='<-' else (rel+"&rarr;")
                out.append("<span class=depedge>" +deparc+ "</span>")
        elif x[0]=='W':
            _,lemma,pos = x
            out.append(lemma)
    return ' '.join(out)
