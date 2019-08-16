import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from components import Column, Header, Row
import config
from auth import auth
from utils import StaticUrlPath

import json
import numpy as np
from copy import deepcopy
import urllib # python 2.7, use urllib.request for python3 instead
from time import time

# def bootstrap_ci(x, n_boot = 5999, interval = .95, func=np.mean):
#     x_boot = []
#     n = x.shape[0]
#     for i_boot in range(n_boot):
#         index = np.random.randint(0,n,n)
#         x_boot += [func(x[index], 0)]
#     lo = np.sort(x_boot,0)[int(n_boot*((1-interval)/2))]
#     hi = np.sort(x_boot,0)[int(n_boot*(1-(1-interval)/2))]
#     return (lo,hi,func(x, 0))

external_css = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'http://www.bic.mni.mcgill.ca/~peterd/dash_demo.css'
]

app = dash.Dash(
    __name__,
    # Serve any files that are available in the `static` folder
    # static_folder='static'
)
app.config['suppress_callback_exceptions']=True
auth(app)
for css in external_css:
    app.css.append_css({"external_url": css})

server = app.server  # Expose the server variable for deployments

tier = 'combined'

# print(urllib.urlopen('http://www.bic.mni.mcgill.ca/~peterd/talk_data_words.json'))

def load_from_url(filename):
    # with urllib.urlopen("http://www.bic.mni.mcgill.ca/~peterd/"+filename) as url:
    #     jdata = json.loads(url.read().decode())
    url = "http://www.bic.mni.mcgill.ca/~peterd/"+filename
    response = urllib.urlopen(url)
    jdata = json.loads(response.read().decode())
    return jdata

a = time()

print("Load data words")
# Load data
# with open('./talk_data_words.json') as f:
#     jdata_w = json.loads(f.read())
# with urllib.request.urlopen("http://www.bic.mni.mcgill.ca/~peterd/talk_data_words.json") as url:
#     jdata_w = json.loads(url.read().decode())
jdata_w = load_from_url('talk_data_words.json')
vocab = jdata_w['vocab']
talknames = [{'label': jj['name'], 'value': ii} for ii, jj in enumerate(jdata_w['talks'])]
print(time()-a)
print("Load data combined")
# with open('./talk_data_combined.json') as f:
#     jdata_cp = json.loads(f.read())
# with urllib.request.urlopen("http://www.bic.mni.mcgill.ca/~peterd/talk_data_combined.json") as url:
#     jdata_w = json.loads(url.read().decode())
jdata_cp = load_from_url('talk_data_combined.json')
jdata_cw = {}
jdata_cw['vocab'] = jdata_cp['vocab_w']
jdata_cw['embed'] = jdata_cp['embed_w']
# jdata_cw['embed'] = jdata_w['embed']
jdata_cw['talks'] = []
for i_talk, talkdat in enumerate(jdata_cp['talks']):
    words_cw = np.array(jdata_w['vocab'])[talkdat['data_words_w']]
    word_start = talkdat['word_start']
    string = []
    for (s,is_w) in zip(words_cw,word_start):
        if is_w:
            string.append(s+' ')
        else:
            string.append(' ')
    jdata_cw['talks'].append({
        'string': string,
        # 'string': talkdat['string'],
        'surprise': talkdat['surprise_w'],
        'entropy': talkdat['entropy_w'],
        'top10words': talkdat['top10words_w'],
        'top10error': talkdat['top10error_w'],
        # 'top10error': np.load('./toperror_w_%d.npy' %i_talk),
        'data_words': talkdat['data_words_w']
    })
def get_jdata(urlcontent, i_tier):
    if urlcontent is not None:
        urlcontent = urlcontent.split('/')[-1]
    if (urlcontent=='combined') or (urlcontent=='meg'):
        jdata = [jdata_cp, jdata_cw][i_tier]
    else:
        jdata = jdata_w
    return jdata
print(time()-a)
print("Load data MEG")
meg_jsonfile = 'main_ridge_subj%d_A+KM+KS+KE+AxKS+AxKE_bandpass_5_0_coreg_True_True_skip1_bandpassam_0_0_model_timeseries.json'
jdata_meg = []
for i_subj in [2]:
    # with open(meg_jsonfile %i_subj) as f:
    #     jdata_meg.append(json.loads(f.read()))
    jdata_meg.append(load_from_url(meg_jsonfile %i_subj))
jdata_meg.append(deepcopy(jdata_meg[-1]))
# for i_talk in range(7):
#     # print([np.shape(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[-1] for i_subj in range(11)])
#     n_samples = min([np.shape(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[-1] for i_subj in range(11)])
#     data = np.array([np.array(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[:,:,:n_samples] for i_subj in range(11)])
#     print(data.shape)
#     # lo, hi, avg = bootstrap_ci(data)
#     avg = np.mean(data,0)
#     jdata_meg[-1]['talks'][i_talk]['y_hat'] = avg
#     # print(np.shape(data))
i_subj = -1
print(time()-a)
print("Done loading data")

n_words_show = 100
i_talk_start = 2
i_word_start = 0
i_word_clicked = 1
n_samples_win = [-100, 400]
t_win = [nn/150 for nn in n_samples_win]
words_show = [i_word_start, i_word_start+n_words_show]
jdict = {'words_show': words_show,
         'word_clicked': i_word_clicked}
jdiv = json.dumps(jdict)

def get_figure_data(jdata, dotsize, word_obs, annot_sel, is_annot):
    x = np.array(jdata['embed'][0])
    y = np.array(jdata['embed'][1])
    def annot_word(i_word, showarrow=True):
        return dict(x=x[i_word+2],
                    y=y[i_word+2],
                    xref='x',
                    yref='y',
                    text=jdata['vocab'][i_word+2],
                    showarrow=showarrow,
                    ax=20,
                    ay=-20)
    if annot_sel=='obs':
        annots = [annot_word(word_obs)]
    elif annot_sel=='none':
        annots = []
    else:
        if len(is_annot)>0:
            annots = [annot_word(i_high) for i_high in is_annot]
        else:
            annots = []
    return {
        'data': [go.Scattergl(
                    x=x,
                    y=y,
                    mode='markers',
                    marker = dict(size = dotsize),
                    text = jdata['vocab']
                 )],
        'layout': go.Layout(
                    autosize=True,
                    hovermode = 'closest',
                    yaxis = dict(zeroline = False, range=[y.min()-5, y.max()+5],
                                 showticklabels=False, mirror=True, showline=True),
                    xaxis = dict(zeroline = False, range=[x.min()-5, x.max()+5],
                                 showticklabels=False, mirror=True, showline=True),
                    showlegend = False,
                    annotations = annots,
                    margin = go.Margin(
                                l=5,
                                r=5,
                                b=15,
                                t=15,
                                pad=4
                             )
                    )
    }


# CSS
container_narrow_style = {
  'margin': '0 auto',
  'max-width': '700px'}

graph_narrow_style = {
  'margin': '0 auto',
  'max-height': '300px'}

text_div = {
  'line-height': '25px',
  'textAlign': 'center',
  'margin': 'auto',
  'vertical-align': 'middle'
}
#
# wrapper = {
#   'text-align': 'center',
#   'line-height': '25px'
# }


def fill_text(jdata, i_talk, metric, i_words, word_clicked):
    # i_words = np.arange(start_word, start_word+n_words_show)
    if metric=='entropy':
        col = '59, 76, 192'
    elif metric=='surprise':
        col = '209, 73, 63'
    else:
        col = '255, 255, 255'
    dat = jdata['talks'][i_talk]
    richcontent = [html.Span('', id='word%d' %i_word, hidden=True) for i_word in range(n_words_show)]
    if metric=='none':
        for i_word, word in enumerate(np.array(dat['string'])[i_words]):
            richcontent[i_word] = html.Span(word.lower(), id='word%d' %i_word)
    else:
        for i_word, (word, val, word_obs) in enumerate(zip(np.array(dat['string'])[i_words],
                                                           np.array(dat[metric])[i_words],
                                                           np.array(dat['data_words'])[i_words])):
            fcolor = 'rgba(0, 0, 0, 1)'
            if i_word==word_clicked:
                bgcolor = 'rgba(0, 0, 0, 1)'
                fcolor = 'rgba(255, 255, 255, 1)'
            else:
                bgcolor = 'rgba(%s, %.2f)' %(col, val/10.)
            if (word_obs<0) and (metric=='surprise'):
                bgcolor = 'rgba(100, 100, 100, .2)'
            richcontent[i_word] = html.Span(word.lower(), id='word%d' %i_word,
                                            style={'background-color': bgcolor,
                                                   'color': fcolor})
    return richcontent

def get_legend(metric):
    # return [html.Span('%d' %ii) for ii in range(0,100,10)]
    return ''

def get_layout(tier='words'):
    meg_title_div = ''
    meg_div = ''
    if (tier=='combined') or (tier=='meg'):
        render_div = [html.Div(fill_text(jdata_cw, 0, 'none', 0, 0), id="render_w", style={'height': '60px', 'padding-top': '10px'}),
                      html.Div(fill_text(jdata_cp, 0, 'none', 0, 0), id="render", style={'height': '140px', 'padding-top': '10px'})]
        title_div = [html.Div('Phone probability', className="four columns"),
                     html.Div('Word probability', className="eight columns")]
        graph_div = [html.Div(dcc.Graph(id='embedding', style={'height': 350}),id='emb',className="four columns"),
                     html.Div(dcc.Graph(id='embedding_w', style={'height': 350}),id='emb_w',className="eight columns")]
        if tier=='meg':
            graph_div = [html.Div(dcc.Graph(id='embedding', style={'height': 175}),id='emb',className="four columns"),
                         html.Div(dcc.Graph(id='embedding_w', style={'height': 175}),id='emb_w',className="eight columns")]
            meg_title_div = [html.Div('Modelled MEG traces', className="twelve columns")]
            meg_div = [html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns'),
                       html.Div(dcc.Graph(id='meg_traces', style={'height': 175}),id='meg',className="ten columns"),
                       html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns')]
    else:
        title_div = [html.Div('Word probability', className="twelve columns")]
        graph_div = [html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns'),
                     html.Div(dcc.Graph(id='embedding', style={'height': 350}),id='emb',className="ten columns"),
                     html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns')]
        render_div = html.Div(fill_text(jdata_cw, 0, 'none', 0, 0), id="render", style={'height': '170px', 'padding-top': '10px'})
    return html.Div(
        [
        html.Div(
            [
             html.Div('Select talk:', className='five columns'),
             html.Div('Color the words by:', className='three columns'),
             html.Div('Word %04d-%04d' %(1, 1+n_words_show), id='wordcount', className='four columns'),
            ],
            className='row', style={'text-align': 'center'}),
        html.Div(
            [
             html.Div(dcc.Dropdown(
                 id="talk-select",
                 options=talknames,
                 value=talknames[i_talk_start]['value']
             ), className='five columns'),
             html.Div(dcc.Dropdown(
                         id="metric-select",
                         options=[
                             {'label': 'Entropy', 'value': 'entropy'},
                             {'label': 'Surprise', 'value': 'surprise'},
                             {'label': 'No coloring', 'value': 'none'}
                         ],
                         value='entropy'
                         ), className='three columns'),
             html.Div(html.Button('<<', id='prev'), className='two columns'),
             html.Div(html.Button('>>', id='next'), className='two columns')
            ], className='row'),
        html.P(render_div, style={'white-space': 'pre-wrap'}),
        html.Div(meg_title_div, className='row', style=text_div),
        html.Div(meg_div, className='row'),
        html.Div(title_div, className = 'row', style=text_div),
        html.Div(graph_div, className = 'row'),
        html.Div(
            [
             html.Div('Select which words to annotate:', className='five columns'),
             html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns'),
             html.Div('Probability cutoff:', className='six columns')
            ], className='row', style={'textAlign': 'center'}),
        html.Div(
            [
             # html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns'),
             html.Div(dcc.Dropdown(
                id="annot-select",
                options=[
                    {'label': 'Spoken', 'value': 'obs'},
                    {'label': 'Predicted', 'value': 'pred'},
                    {'label': 'No annotations', 'value': 'none'},
                ],
                value='obs'
             ), className='five columns'),
             html.Div(html.P('   ', style={'white-space': 'pre-wrap'}), className='one columns'),
             html.Div(dcc.Slider(
                id='prob-cutoff',
                min=-3.2,
                max=-.1,
                step=.01,
                value=-1.85,
                marks={i: 'p > 10^{}'.format(i) for i in [-3, -2, -1]},
                disabled=True
             ), className='six columns')
             # html.Div(html.P(' ', style={'white-space': 'pre-wrap'}), className='one columns')
            ], className='row'),
        ], id="container", style=container_narrow_style
    )

# Standard Dash app code below
app.layout = html.Div([
    # html.H1('Tedlium RNN demo', style={'textAlign': 'center'}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div("start", id="output", hidden=True),
    html.Div(jdiv, id="divdata", hidden=True)
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def change_page_content(urlcontent):
    if urlcontent is not None:
        urlcontent = urlcontent.split('/')[-1]
    if urlcontent=='combined':
        return get_layout('combined')
    elif urlcontent=='meg':
        return get_layout('meg')
    else:
        return get_layout('words')

def get_words_show(jdata, n_pre, n_nex, i_talk):
    if n_nex is None:
        n_nex = 0
    if n_pre is None:
        n_pre = 0
    i_word = (n_nex-n_pre)*n_words_show
    i_word_end = i_word+n_words_show
    n_words_talk = len(jdata['talks'][i_talk]['string'])
    if i_word < 0:
        i_word = 0
        i_word_end = i_word+n_words_show
    elif (n_words_talk < i_word_end) and (n_words_talk >= i_word):
        i_word_end = n_words_talk
    elif n_words_talk < i_word:
        i_word = int(np.floor(n_words_talk/n_words_show)*n_words_show)
        i_word_end = n_words_talk
    return [i_word, i_word_end]

@app.callback(Output('divdata', 'children'),
              [Input('word%d' %i_word, 'n_clicks') for i_word in range(n_words_show)]+\
              [Input('prev', 'n_clicks'), Input('next', 'n_clicks')],
              [State('divdata', 'children'), State('talk-select', 'value'),
               State('url', 'pathname')])
def update_divdata(*args):
    urlcontent = args[-1]
    i_talk = args[-2]
    jdiv = args[-3]
    n_nex = args[-4]
    n_pre = args[-5]
    word_clicks = args[:-6]
    jdata = get_jdata(urlcontent, 0)
    divdict = json.loads(jdiv)
    # words to show
    divdict['words_show'] = get_words_show(jdata, n_pre, n_nex, i_talk)
    # how often have words been clicked
    word_clicks = [a if a!=None else 0 for a in word_clicks]
    clicked = np.where(word_clicks)[0]
    divdict['word_clicked'] = int(clicked[0] if len(clicked)>0 else i_word_clicked)
    return json.dumps(divdict)

@app.callback(Output('wordcount', 'children'), [Input('divdata', 'children')])
def update_wordcount(jdiv):
    wc = json.loads(jdiv)['words_show']
    return 'Word %04d-%04d' %(wc[0]+1, wc[1]+1)

# Render talk transcript with Entropy or Surprise coloring
@app.callback(Output('render', 'children'), [Input('talk-select', 'value'),
                                             Input('metric-select', 'value'),
                                             Input('divdata', 'children'),
                                             Input('url', 'pathname')])
def update_content(i_talk, metric, jdiv, urlcontent):
    jdata = get_jdata(urlcontent, 0)
    divdata = json.loads(jdiv)
    wc = divdata['words_show']
    word_clicked = divdata['word_clicked']
    i_words = np.arange(wc[0], wc[1])
    richcontent = fill_text(jdata, i_talk, metric, i_words, word_clicked)
    return richcontent

@app.callback(Output('render_w', 'children'), [Input('talk-select', 'value'),
                                               Input('metric-select', 'value'),
                                               Input('divdata', 'children'),
                                               Input('url', 'pathname')])
def update_content(i_talk, metric, jdiv, urlcontent):
    jdata = get_jdata(urlcontent, 1)
    divdata = json.loads(jdiv)
    wc = divdata['words_show']
    word_clicked = divdata['word_clicked']
    i_words = np.arange(wc[0], wc[1])
    richcontent = fill_text(jdata, i_talk, metric, i_words, word_clicked)
    return richcontent

def update_figure_with_jdata(jdata, jdiv, i_talk, annot_sel, prob_cutoff):
    divdata = json.loads(jdiv)
    i_clicked = divdata['word_clicked']+divdata['words_show'][0]
    # Calculate dot sizes
    dotsizes = np.ones(len(jdata['vocab']))
    dat = jdata['talks'][i_talk]
    i_words = np.array(dat['top10words'],dtype=int)[i_clicked,:]
    errs = np.array(dat['top10error'])[i_clicked,:]
    is_annot = np.exp(-errs)>(10**prob_cutoff)
    dotsize = np.max(np.array([np.ones(len(errs))*3,13-errs]),0)
    dotsizes[i_words] = dotsize
    return get_figure_data(jdata, dotsizes, dat['data_words'][i_clicked],
                           annot_sel, i_words[is_annot])

# Update the embedding figure after word is clicked
@app.callback(Output('embedding', 'figure'), [Input('divdata', 'children'),
                                              Input('talk-select', 'value'),
                                              Input('annot-select', 'value'),
                                              Input('prob-cutoff', 'value'),
                                              Input('url', 'pathname')])
def update_figure(jdiv, i_talk, annot_sel, prob_cutoff, urlcontent):
    jdata = get_jdata(urlcontent, 0)
    return update_figure_with_jdata(jdata, jdiv, i_talk, annot_sel, prob_cutoff)

# Update the embedding figure after word is clicked
@app.callback(Output('embedding_w', 'figure'), [Input('divdata', 'children'),
                                                Input('talk-select', 'value'),
                                                Input('annot-select', 'value'),
                                                Input('prob-cutoff', 'value'),
                                                Input('url', 'pathname')])
def update_figure(jdiv, i_talk, annot_sel, prob_cutoff, urlcontent):
    jdata = get_jdata(urlcontent, 1)
    return update_figure_with_jdata(jdata, jdiv, i_talk, annot_sel, prob_cutoff)

# Update the embedding figure after word is clicked
@app.callback(Output('meg_traces', 'figure'), [Input('divdata', 'children'),
                                               Input('talk-select', 'value'),
                                               Input('metric-select', 'value'),
                                               Input('url', 'pathname')])
def update_figure_meg(jdiv, i_talk, metric, urlcontent):
    if urlcontent is not None:
        urlcontent = urlcontent.split('/')[-1]
    if urlcontent=='meg':
        # print(jdata_meg['talks'][i_talk]['y_hat'][0][:20])
        divdata = json.loads(jdiv)
        i_clicked = divdata['word_clicked']+divdata['words_show'][0]
        # print(jdata_meg[i_subj]['talks'][i_talk]['index_words'][i_clicked])
        index_words = np.array(jdata_meg[i_subj]['talks'][i_talk]['index_words'])
        i_sample = index_words[i_clicked]
        samples_win = [i_sample+n_samples_win[0], i_sample+n_samples_win[1]]
        samples_show = np.arange(samples_win[0], samples_win[1])
        index_words_show = index_words[(index_words>samples_win[0])&(index_words<samples_win[1])]
        strings = np.array(jdata_cw['talks'][i_talk]['string'])[(index_words>samples_win[0])&(index_words<samples_win[1])]
        # print(x[index_words_show-samples_win[0]])
        meg_traces_base = np.array(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[-1,:,samples_show].T
        meg_traces_full = np.array(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[0,:,samples_show].T
        if metric=='entropy':
            col = '59, 76, 192'
            meg_traces = np.array(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[2,:,samples_show].T
        elif metric=='surprise':
            col = '209, 73, 63'
            meg_traces = np.array(jdata_meg[i_subj]['talks'][i_talk]['y_hat'])[1,:,samples_show].T
        else:
            col = '100, 100, 100'
            meg_traces = meg_traces_base
        x = np.linspace(t_win[0],t_win[1],len(meg_traces[0]))
    return {
        'data': [go.Scatter(
                    x=x,
                    y=meg_trace-3+i_trace*6,
                    hoverinfo='none',
                    line = dict(
                        color = ('rgba(100,100,100,.75)'))
                 ) for i_trace, meg_trace in enumerate(meg_traces_base)] + [go.Scatter(
                 #             x=x,
                 #             y=meg_trace-3+i_trace*6,
                 #             hoverinfo='none',
                 #             line = dict(
                 #                 color = ('rgba(30,30,30,.75)'))
                 # ) for i_trace, meg_trace in enumerate(meg_traces_full)] + [go.Scatter(
                    x=x,
                    y=meg_trace-3+i_trace*6,
                    hoverinfo='none',
                    line = dict(
                        color = ('rgba(%s,.75)' %col))
                 ) for i_trace, meg_trace in enumerate(meg_traces)] + [go.Scatter(
                             x=x,
                             y=np.zeros(len(meg_trace))-3+i_trace*6,
                             hoverinfo='none',
                             line = dict(
                                 color = ('rgba(150,150,150,.75)'),
                                 width = 1)
                 ) for i_trace, meg_trace in enumerate(meg_traces)] + [go.Scatter(
                             x=x[index_words_show-samples_win[0]],
                             y=np.zeros(len(index_words_show)),
                             text=strings,
                             showlegend=False,
                             textposition='bottom center',
                             mode='markers',
                             marker = dict(size = 2,
                                           color = ('rgba(50,50,50,.75)'))
                 )],
        'layout': go.Layout(
                    autosize=True,
                    hovermode = 'closest',
                    yaxis = dict(zeroline = False, range=[-6.5, 6.5],
                                 showticklabels=False, mirror=True, showline=True),
                    xaxis = dict(range=t_win,
                                 mirror=True, showline=True),
                    showlegend = False,
                    margin = go.Margin(
                                l=5,
                                r=5,
                                b=30,
                                t=15,
                                pad=4
                             )
                    )
    }

@app.callback(Output('prob-cutoff', 'disabled'), [Input('annot-select', 'value')])
def disable_num_annot(annot_sel):
    if annot_sel=='pred':
        return False
    else:
        return True

# # Optionally include CSS
# app.css.append_css({
#     'external_url': [
#         StaticUrlPath(css) for css in [
#             'dash.css', 'grid.css', 'loading.css', 'page.css',
#             'spacing.css', 'styles.css', 'tables.css', 'typography.css'
#         ]
#     ]
# })

if __name__ == '__main__':
    app.run_server(debug=True)
