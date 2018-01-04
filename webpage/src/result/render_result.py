import os
import re
import argparse
import logging
import html


parser = argparse.ArgumentParser(description='Renders one text file in HTML')
parser.add_argument('fname', type=str, nargs='+',
                    help='The text file to render')
parser.add_argument('--limit', type=int, default=100,
                    help='Max number of samples to show')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)
debug = logger.debug


PAGE = """<html>
<head>
<meta charset="UTF-8">
<meta author="Zhitao Gong">
<meta description="Adversarial text results demo">
<meta keywords="adversarial,text,tensorflow,python,deep learning,security">
<link rel="stylesheet" type="text/css" href="result.css">
<title>Adversarial Text Demo - {dataset} - {method}</title>
</head>
<body>
<main>
<h1>Adversarials on dataset <span class="dataset">{dataset}</span> via
<span class="method">{method}</span></h1>
<p>Changed words are <span class="hot">highlighted</span>. WMD refers to Word
Mover's Distance.  The number in parenthesis following WMD is the change rate,
i.e., number of changes divided by sentence length.</p>
<div class="results">{result}</div>
</main>
</body>
</html>
"""

SAMPLE = """<div class="sample">
<div class="clean">{clean}</div>
<div class="label">{label}</div>
<div class="wmd">{wmd}</div>
<div class="adversarial">{adver}</div>
</div>"""


def txt2html(ifname, ofname):
    debug('{0} --> {1}'.format(ifname, ofname))

    p0 = re.compile(r'\(\(\(([^)]*)\)\)\) ')
    p1 = re.compile(r'\[\[\[([^]]*)\]\]\]')
    repl = r'<span class="hot">\1</span>'
    pad = re.compile(r'<pad>')

    def _format_label(label):
        if '-' in label:
            # multi-label case, in [a-b] format
            ret = label[1:-1]   # remove []
            ret = ret.replace('-', ' &rarr; ')
        else:
            # binary case
            org = int(label)
            adv = 1 - org
            ret = '{0} &rarr; {1}'.format(org, adv)
        return ret

    out = [SAMPLE.format(clean="Clean Text", label="Label", wmd="WMD",
                         adver="Adversarial Text")]
    with open(ifname, 'r') as f:
        for i, line in enumerate(f):
            label, wmd, n, per, text = line.split(maxsplit=4)

            text = re.sub(pad, '', text)
            text = text.strip()
            text = html.escape(text)

            clean = re.sub(p0, '', text)
            clean = re.sub(p1, repl, clean)
            adver = re.sub(p1, '', text)
            adver = re.sub(p0, repl, adver)

            label = _format_label(label)
            wmd = '{0} ({1})'.format(wmd, per)

            cur = SAMPLE.format(clean=clean, label=label, wmd=wmd, adver=adver)
            out.append(cur)

            if i >= args.limit:
                break

    result = ''.join(out)
    dataset, method, _ = os.path.basename(ifname).split('_', maxsplit=2)
    page = PAGE.format(dataset=dataset, method=method, result=result)
    with open(ofname, 'w') as f:
        f.write(page)


for f in args.fname:
    ifname = os.path.expanduser(f)
    ofname = os.path.splitext(os.path.basename(ifname))[0] + '.html'
    txt2html(ifname, ofname)
