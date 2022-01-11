import plotly.graph_objects as go
import numpy as np

def Histogram2D_Hot(filename, data, SNR=0, EVM=0, bercount=(0, 0, 0), path=None):
    x = np.real(data)
    y = np.imag(data)
    miny = y.min()
    fig = go.Figure()
    filename = str(filename)
    # fig.add_trace(go.Histogram2dContour(
    #     x=x,
    #     y=y,
    #     colorscale='Hot',
    #     reversescale=True,
    #     xaxis='x',
    #     yaxis='y'
    # ))
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y,
    #     xaxis='x',
    #     yaxis='y',
    #     mode='markers',
    #     marker=dict(
    #         color='rgba(255,156,0,1)',
    #         size=3)
    # ))
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='#F58518'
        )
    ))
    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='#F58518'
        )
    ))
    # if SNR != 0:
    #     fig.add_annotation(
    #         text='SNR:{:.2f}(dB)<br>EVM:{:.2f}(%)<br>Bercount:{:.2E} [{}/{}]'.format(SNR, EVM * 100, bercount[0],
    #                                                                                        bercount[1], bercount[2]),
    #         align='left',
    #         showarrow=False,
    #         font_family='Arial',
    #         font_size=17,
    #         font_color='white',
    #         bgcolor='black',
    #         # xref='x2',
    #         x=0,
    #         y=miny - 0.3,
    #         bordercolor='orange',
    #         borderwidth=5
    #         )
        # fig.add_annotation(
        #     x=0,
        #     y=miny-0.3,
        #     text="SNR = {:.2f}(dB)".format(SNR),
        #     showarrow=False)
    fig.add_trace(go.Histogram2d(
        x=x,
        y=y,
        colorscale='Hot',
        nbinsx=256,
        nbinsy=256,
        zauto=True
    ))
    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.9],
            showgrid=False,
            fixedrange=True,
            title="In-Phase",
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.9],
            showgrid=False,
            fixedrange=True,
            title="Quadrature-Phase",
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            showgrid=False,
            fixedrange=True
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.905, 1],
            showgrid=False,
            fixedrange=True
        ),
        height=800,
        width=800,
        bargap=0,
        hovermode='closest',
        showlegend=False,
        title=go.layout.Title(text="Color Histogram---" + filename),
        font=dict(
            family="Arial",
            size=20,
            color="Black"),
        # yaxis_range=[-4, 4],
        # xaxis_range=[-4,4]
        )
    if SNR != 0:
        print("SNR = {:.2f}(dB)".format(SNR))
        print("EVM = {:.2f}(%)".format(EVM))
        # print("bercount = {:.2E}".format(bercount[0]))
        print("bercount = {:.2E}".format(bercount))
        fig.update_layout(
            xaxis=dict(
                zeroline=False,
                domain=[0, 0.9],
                showgrid=False,
                fixedrange=True,
                # title='In-Phase<br>SNR:{:.2f}(dB) || EVM:{:.2f}(%) || Bercount:{:.2E} [{}/{}]'.format(SNR, EVM * 100, bercount[0],bercount[1], bercount[2])
                title='In-Phase<br>SNR:{:.2f}(dB) || EVM:{:.2f}(%) || Bercount:{:.2E}'.format(SNR, EVM * 100, bercount)
            ),
            font=dict(
                family="Arial",
                size=20,
                color="Black"),
        )
    if path:
        fig.write_image(path+"/{}.png".format(filename))
    else:
        fig.write_image("/Users/suanyu/Documents/DSP/coherent/16QAM/image/{}.png".format(filename))