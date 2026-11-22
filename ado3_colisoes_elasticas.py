"""
ADO-3 - Física: Práticas Extensivas
Opção A: Colisões elásticas unidimensionais (tema: Pêndulo de Newton)
Grupo: Maria Clara Fernandes Bessa, Nycolly Vitória de Oliveira, João Victor Tavares de Andrade
Somente fórmulas fechadas (sem integração numérica). Unidades do SI.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.widgets import Slider, Button, RadioButtons

# física/contas fechadas (funções separadas da interface)
def momento(m1, v1, m2, v2):
    """Momento linear total p = m1*v1 + m2*v2  [kg.m/s]"""
    return m1 * v1 + m2 * v2

def cinetica(m1, v1, m2, v2):
    """Energia cinética total K = 1/2 m1 v1^2 + 1/2 m2 v2^2  [J]"""
    return 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2

def colisao_elastica(m1, m2, v1, v2):
    """Velocidades finais de uma colisão elástica 1D (fórmulas fechadas)"""
    u1 = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    u2 = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return u1, u2

def impulsos(m1, m2, v1, v2, u1, u2, dt):
    """Impulsos J1, J2 [N.s] e força média Fmed = |J|/dt [N]"""
    J1 = m1 * (u1 - v1)
    J2 = m2 * (u2 - v2)
    Fmed = abs(J1) / dt
    return J1, J2, Fmed


def instante_colisao(x1, x2, v1, v2, meia_soma):
    """
    Instante tc em que a distância entre os centros = soma das meias-larguras
    x2 - x1 - meia_soma = (v1 - v2) * tc   -> exige v1 > v2
    Retorna None se não houver colisão
    """
    if v1 <= v2:
        return None
    return (x2 - x1 - meia_soma) / (v1 - v2)


# casos predefinidos (m1, m2, v1)
CASOS = {
    "Caso 1: m1 = m2":  (1.0, 1.0, 4.0),
    "Caso 2: m1 < m2":  (1.0, 4.0, 5.0),
    "Caso 3: m1 > m2":  (4.0, 1.0, 5.0),
}


# estado e constantes
estado = {"t": 0.0, "pausado": False, "ignorar_eventos": False, "n": 0,
          "pronto": False, "redesenhar": False, "ultimo": None}

DT_MAX = 0.1         # maior avanço de tempo permitido entre dois quadros [s]
VEL_ANIM = 0.7       # velocidade da animação (1 = tempo real; menor = câmera lenta)
X1_0 = -4.0          # posição inicial do centro do bloco 1 [m]
X2_0 = 0.0           # posição inicial do centro do bloco 2 [m]
T_APOS_CHOQUE = 3.0  # tempo mostrado depois do choque antes de reiniciar [s]
XMIN, XMAX = -8.0, 16.0   # limites visíveis do eixo x [m]
YMIN, YMAX = -0.9, 3.75   # limites visíveis do eixo y (chão em y < 0)
ESC_SETA = 0.25      # comprimento da seta de velocidade [m por m/s]


# tema visual (só estética): troque entre "escuro" e "claro"
TEMA = "escuro"

PALETAS = {
    "claro": dict(
        fundo="#EEF1F6", cartao="#FFFFFF", borda="#D9DEE8", texto="#1F2937",
        suave="#6B7280", grade="#E5E9F2", ceu_topo="#DCE8FB", ceu_base="#F6F9FE",
        chao="#DDE3EC", linha_chao="#9AA5B5", ok="#15803D", erro="#B91C1C",
        aviso_fc="#FEE2E2", aviso_ec="#FCA5A5", bloco1="#2F6FED",
        bloco1_borda="#1B47A8", bloco2="#E5484D", bloco2_borda="#A62B30",
        destaque="#2F6FED", destaque_hover="#1E55C4", botao2="#E8EDF7",
        botao2_hover="#D6DEEE", continuar="#16A34A", continuar_hover="#15803D",
        sombra=0.07),
    "escuro": dict(
        fundo="#0D1320", cartao="#172033", borda="#2B3954", texto="#E8EDF6",
        suave="#8E9BB2", grade="#27344D", ceu_topo="#26396B", ceu_base="#121B30",
        chao="#1D283E", linha_chao="#51628A", ok="#4ADE80", erro="#FB7185",
        aviso_fc="#3B1A22", aviso_ec="#8A2F3E", bloco1="#4C8DFF",
        bloco1_borda="#A9C8FF", bloco2="#FF6B6B", bloco2_borda="#FFB4B4",
        destaque="#4C8DFF", destaque_hover="#6BA2FF", botao2="#27344D",
        botao2_hover="#34456A", continuar="#22A262", continuar_hover="#2EBD75",
        sombra=0.35),
}
P = PALETAS[TEMA]

COR1, COR2 = P["bloco1"], P["bloco2"]
COR1_BORDA, COR2_BORDA = P["bloco1_borda"], P["bloco2_borda"]
COR_FUNDO, COR_CARTAO, COR_BORDA = P["fundo"], P["cartao"], P["borda"]
COR_TEXTO, COR_SUAVE, COR_GRADE = P["texto"], P["suave"], P["grade"]
COR_OK, COR_ERRO, COR_DESTAQUE = P["ok"], P["erro"], P["destaque"]


def largura(m):
    """Tamanho visual do bloco (largura = altura) dependente da massa."""
    return 1.0 * m ** (1 / 3) + 0.45


# interface: layout em grade, medido em polegadas (assim os espaços entre cartões e as margens internas ficam iguais)
plt.rcParams["toolbar"] = "None"   # sem barra de ferramentas (ficava lá em baixo)
LARG_FIG, ALT_FIG = 13.6, 6.6      # proporção parecida com a de telas de notebook
fig = plt.figure(figsize=(LARG_FIG, ALT_FIG), facecolor=COR_FUNDO)
try:
    fig.canvas.manager.set_window_title("ADO-3 · Colisões elásticas 1D")
except Exception:
    pass


def fr(x, y, w, h):
    """Converte (x, y, largura, altura) de polegadas para frações da janela."""
    return [x / LARG_FIG, y / ALT_FIG, w / LARG_FIG, h / ALT_FIG]


# medidas da grade [polegadas]
M_LAT, M_BAIXO = 0.25, 0.17     # margens laterais e inferior
G = 0.14                        # espaço entre cartões (igual em todo lugar)
ALT_TOPO = 0.42                 # faixa do título principal
W_DIR = 3.50                    # largura da coluna da direita
W_ESQ = LARG_FIG - 2 * M_LAT - G - W_DIR
X_ESQ, X_DIR = M_LAT, M_LAT + W_ESQ + G
Y_TOPO = ALT_FIG - 0.07 - ALT_TOPO          # topo dos cartões

H_CTRL, H_GRAF = 1.42, 1.98
H_ANIM = Y_TOPO - M_BAIXO - H_CTRL - H_GRAF - 2 * G
H_LIVE = 1.05
H_RES = Y_TOPO - M_BAIXO - H_CTRL - H_LIVE - 2 * G
W_GRAF = (W_ESQ - G) / 2

# largura aproximada do rótulo de velocidade, em metros do eixo x
# (usada para reiniciar antes de o rótulo sair do gráfico)
ROTULO_M = 0.95 * (XMAX - XMIN) / (W_ESQ - 0.60)

# (x, y, largura, altura) de cada cartão
y_ctrl = M_BAIXO
y_graf = y_ctrl + H_CTRL + G
y_anim = y_graf + H_GRAF + G
y_live = y_ctrl + H_CTRL + G
y_res = y_live + H_LIVE + G
CT_ANIM = (X_ESQ, y_anim, W_ESQ, H_ANIM)
CT_GP = (X_ESQ, y_graf, W_GRAF, H_GRAF)
CT_GK = (X_ESQ + W_GRAF + G, y_graf, W_GRAF, H_GRAF)
CT_CTRL = (X_ESQ, y_ctrl, W_ESQ, H_CTRL)
CT_RES = (X_DIR, y_res, W_DIR, H_RES)
CT_LIVE = (X_DIR, y_live, W_DIR, H_LIVE)
CT_CASOS = (X_DIR, y_ctrl, W_DIR, H_CTRL)


def cartao(caixa):
    """Cartão de cantos arredondados, com sombra leve, atrás dos elementos."""
    x, y, w, h = [v for v in fr(*caixa)]
    estilo = dict(boxstyle=f"round,pad=0,rounding_size={0.12 / LARG_FIG}",
                  mutation_aspect=LARG_FIG / ALT_FIG, transform=fig.transFigure)
    fig.add_artist(FancyBboxPatch((x + 0.0012, y - 0.0025), w, h, fc="black",
                                  ec="none", alpha=P["sombra"], zorder=-20,
                                  **estilo))
    fig.add_artist(FancyBboxPatch((x, y), w, h, fc=COR_CARTAO, ec=COR_BORDA,
                                  lw=1, zorder=-10, **estilo))


def rotulo_cartao(caixa, texto, cor):
    """Título pequeno no topo de um cartão, com uma barrinha colorida."""
    x, y, w, h = caixa
    yc = (y + h - 0.17) / ALT_FIG
    fig.add_artist(Rectangle(((x + 0.14) / LARG_FIG, yc - 0.075 / ALT_FIG),
                             0.05 / LARG_FIG, 0.15 / ALT_FIG,
                             transform=fig.transFigure, fc=cor, ec="none",
                             zorder=-5))
    fig.text((x + 0.28) / LARG_FIG, yc, texto, fontsize=8.5,
             fontweight="bold", color=COR_SUAVE, va="center")


for ct in (CT_ANIM, CT_GP, CT_GK, CT_CTRL, CT_RES, CT_LIVE, CT_CASOS):
    cartao(ct)

# faixa do título principal (dois quadradinhos = os dois blocos)
y_titulo = (ALT_FIG - 0.07 - ALT_TOPO / 2) / ALT_FIG
fig.text(M_LAT / LARG_FIG, y_titulo, "■", fontsize=17, color=COR1, va="center")
fig.text((M_LAT + 0.26) / LARG_FIG, y_titulo, "■", fontsize=17, color=COR2,
         va="center")
fig.text((M_LAT + 0.62) / LARG_FIG, y_titulo, "Colisões elásticas 1D",
         fontsize=19, fontweight="bold", color=COR_TEXTO, va="center")
fig.text((LARG_FIG - M_LAT) / LARG_FIG, y_titulo,
         "ADO-3 · Colisões, Impulso e Conservação do Momento Linear · Pêndulo de Newton (Opção A)",
         fontsize=10, color=COR_SUAVE, ha="right", va="center")

rotulo_cartao(CT_ANIM, "SIMULAÇÃO", COR_DESTAQUE)
rotulo_cartao(CT_RES, "RESULTADOS (SI)", "#34D399")
rotulo_cartao(CT_LIVE, "AGORA (TEMPO REAL)", "#FBBF24")
rotulo_cartao(CT_CTRL, "CONTROLES", "#A78BFA")
rotulo_cartao(CT_CASOS, "CASOS PRÉ-DEFINIDOS (v₂ = 0)", "#2DD4BF")

# -- área da animação --
ax = fig.add_axes(fr(CT_ANIM[0] + 0.30, CT_ANIM[1] + 0.50,
                     CT_ANIM[2] - 0.60, CT_ANIM[3] - 0.50 - 0.36))
# céu em degradê (imagem estática de fundo)
topo, base = np.array(to_rgb(P["ceu_topo"])), np.array(to_rgb(P["ceu_base"]))
grad = np.linspace(0, 1, 64).reshape(-1, 1, 1)
ax.imshow(topo + (base - topo) * grad, extent=(XMIN, XMAX, YMIN, YMAX),
          aspect="auto", interpolation="bilinear", zorder=0)
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_xlabel("posição x [m]", fontsize=8, labelpad=6, color=COR_SUAVE)
ax.tick_params(labelsize=8.5, length=3, color=COR_BORDA, labelcolor=COR_SUAVE)
ax.set_yticks([])
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)
ax.spines["bottom"].set_color(COR_BORDA)
ax.add_patch(Rectangle((XMIN, YMIN), XMAX - XMIN, -YMIN, fc=P["chao"],
                       ec="none", zorder=1))                         # chão
ax.axhline(0, color=P["linha_chao"], lw=1.8, zorder=2)
for xg in range(-5, 16, 5):                                          # marcas
    ax.plot([xg, xg], [YMIN, 0], color=COR_GRADE, lw=1, zorder=2)

bloco1 = Rectangle((0, 0), 1, 1, fc=COR1, ec=COR1_BORDA, lw=1.5, zorder=3)
bloco2 = Rectangle((0, 0), 1, 1, fc=COR2, ec=COR2_BORDA, lw=1.5, zorder=3)
ax.add_patch(bloco1)
ax.add_patch(bloco2)

kw_rotulo = dict(ha="center", va="center", color="white", fontweight="bold",
                 fontsize=8, zorder=4)
caixa_v = dict(fc=P["ceu_base"], ec="none", pad=1.5, alpha=0.75)
txt_b1 = ax.text(0, 0, "", **kw_rotulo)
txt_b2 = ax.text(0, 0, "", **kw_rotulo)
txt_v1 = ax.text(0, 0, "", ha="right", va="bottom", fontsize=9.5,
                 fontweight="bold", color=COR1, bbox=caixa_v, zorder=4, clip_on=True)
txt_v2 = ax.text(0, 0, "", ha="left", va="bottom", fontsize=9.5,
                 fontweight="bold", color=COR2, bbox=caixa_v, zorder=4, clip_on=True)

# setas de velocidade (comprimento proporcional a v)
seta1 = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=COR1, lw=2.2,
                                    mutation_scale=13, shrinkA=0, shrinkB=0))
seta2 = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=COR2, lw=2.2,
                                    mutation_scale=13, shrinkA=0, shrinkB=0))

# marca do ponto de choque (a faixa do chão serve de "linha de status")
Y_FAIXA = YMIN / 2
linha_choque = ax.axvline(0, color=COR_SUAVE, ls="--", lw=1, visible=False,
                          zorder=2)
txt_choque = ax.text(0, Y_FAIXA, "", ha="left", va="center", fontsize=8,
                     color=COR_SUAVE, visible=False, zorder=4)

txt_aviso = ax.text(0.5, (Y_FAIXA - YMIN) / (YMAX - YMIN), "",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=COR_ERRO, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc=P["aviso_fc"],
                              ec=P["aviso_ec"]))

# painel de resultados (seções posicionadas em polegadas, a partir do topo)
H_TXT = CT_RES[3] - 0.38 - 0.14
ax_txt = fig.add_axes(fr(CT_RES[0] + 0.28, CT_RES[1] + 0.14, W_DIR - 0.56, H_TXT))
ax_txt.axis("off")
kw_corpo = dict(va="top", ha="left", family="monospace", fontsize=9,
                color=COR_TEXTO, linespacing=1.3)
kw_titulo = dict(va="top", ha="left", fontsize=8, fontweight="bold",
                 color=COR_SUAVE)
kw_selo = dict(va="top", ha="right", fontsize=8.5, fontweight="bold")


def yt(pol):
    """Posição vertical no painel: polegadas a partir do topo -> fração do eixo."""
    return 1 - pol / H_TXT


def secao(topo_pol, titulo):
    """Cria o título de uma seção e devolve o texto do corpo logo abaixo."""
    t = ax_txt.text(0, yt(topo_pol), titulo, **kw_titulo)
    corpo = ax_txt.text(0, yt(topo_pol + 0.165), "", **kw_corpo)
    return t, corpo


T1, T2, T3, T4 = 0.0, 0.71, 1.255, 1.965     # topo de cada seção [pol]
t_cv, txt_cv = secao(T1, "CHOQUE E VELOCIDADES (s · m/s)")
t_p, txt_p = secao(T2, "MOMENTO LINEAR (kg·m/s)")
t_k, txt_k = secao(T3, "ENERGIA CINÉTICA (J)")
t_j, txt_j = secao(T4, "IMPULSO (N·s) E FORÇA MÉDIA (N)")
selo_p = ax_txt.text(0.92, yt(T2), "", **kw_selo)   # selo "✓ conservado"
selo_k = ax_txt.text(0.69, yt(T3), "", **kw_selo)   # selo "✓ ΔK = 0"
secoes_res = [t_cv, txt_cv, t_p, txt_p, selo_p, t_k, txt_k, selo_k, t_j, txt_j]

# mensagem do caso sem colisão (R5), escondida enquanto há colisão
txt_sem_t = ax_txt.text(0, yt(0), "SEM COLISÃO", va="top", ha="left",
                        fontsize=13, fontweight="bold", color=COR_ERRO,
                        visible=False)
txt_sem = ax_txt.text(0, yt(0.42), "", visible=False, **kw_corpo)

# estado ao vivo (muda a cada quadro; fica em eixo próprio por causa do blit)
ax_status = fig.add_axes(fr(CT_LIVE[0] + 0.28, CT_LIVE[1] + 0.12,
                            W_DIR - 0.56, CT_LIVE[3] - 0.12 - 0.40))
ax_status.axis("off")
txt_status = ax_status.text(0, 1, "", va="top", ha="left", family="monospace",
                            fontsize=9, color=COR_TEXTO, linespacing=1.3)

# gráficos de barras (empilhadas por bloco)
ax_p = fig.add_axes(fr(CT_GP[0] + 0.55, CT_GP[1] + 0.47, W_GRAF - 0.55 - 0.25,
                       H_GRAF - 0.47 - 0.42))
ax_k = fig.add_axes(fr(CT_GK[0] + 0.55, CT_GK[1] + 0.47, W_GRAF - 0.55 - 0.25,
                       H_GRAF - 0.47 - 0.42))


# sliders (5 linhas, espaçamento fixo)
def criar_slider(linha, rotulo, vmin, vmax, vinit, passo, formato):
    y_centro = CT_CTRL[1] + CT_CTRL[3] - 0.47 - linha * 0.185
    eixo = fig.add_axes(fr(CT_CTRL[0] + 1.0, y_centro - 0.06, 3.6, 0.12),
                        facecolor="none")
    for sp in eixo.spines.values():
        sp.set_visible(False)
    sl = Slider(eixo, rotulo, vmin, vmax, valinit=vinit, valstep=passo,
                valfmt=formato, color=COR_DESTAQUE, track_color=COR_GRADE,
                initcolor="none",
                handle_style={"facecolor": "white", "edgecolor": COR_DESTAQUE,
                              "size": 13})
    sl.label.set_fontsize(10)
    sl.label.set_color(COR_TEXTO)
    sl.valtext.set_fontsize(10)
    sl.valtext.set_color(COR_TEXTO)
    sl.valtext.set_x(1.05)   # afasta o número da bolinha no valor máximo
    return sl


s_m1 = criar_slider(0, r"$m_1$ [kg]", 0.5, 10.0, 1.0, 0.1, "%.1f")
s_m2 = criar_slider(1, r"$m_2$ [kg]", 0.5, 10.0, 1.0, 0.1, "%.1f")
s_v1 = criar_slider(2, r"$v_1$ [m/s]", -10.0, 10.0, 4.0, 0.1, "%.1f")
s_v2 = criar_slider(3, r"$v_2$ [m/s]", -10.0, 10.0, 0.0, 0.1, "%.1f")
s_dt = criar_slider(4, r"$\Delta t$ [ms]", 1.0, 100.0, 10.0, 1.0, "%d")

# botões (lado a lado)
y_bot = CT_CTRL[1] + 0.54
ax_pause = fig.add_axes(fr(CT_CTRL[0] + 5.75, y_bot, 1.65, 0.42))
ax_reset = fig.add_axes(fr(CT_CTRL[0] + 7.60, y_bot, 1.65, 0.42))
for eixo_b in (ax_pause, ax_reset):
    for sp in eixo_b.spines.values():
        sp.set_visible(False)
b_pause = Button(ax_pause, "Pausar", color=COR_DESTAQUE,
                 hovercolor=P["destaque_hover"])
b_pause.label.set_color("white")
b_pause.label.set_fontweight("bold")
b_pause.label.set_fontsize(10.5)
b_reset = Button(ax_reset, "Reiniciar", color=COR_DESTAQUE,
                 hovercolor=P["destaque_hover"])
b_reset.label.set_color("white")
b_reset.label.set_fontweight("bold")
b_reset.label.set_fontsize(10.5)

# menu de casos
ax_radio = fig.add_axes(fr(CT_CASOS[0] + 0.30, CT_CASOS[1] + 0.12,
                           W_DIR - 0.60, CT_CASOS[3] - 0.12 - 0.42),
                        facecolor="none")
for sp in ax_radio.spines.values():
    sp.set_visible(False)
radio = RadioButtons(ax_radio, list(CASOS.keys()), active=0,
                     activecolor=COR_DESTAQUE)
for r in radio.labels:
    r.set_fontsize(10.5)
    r.set_color(COR_TEXTO)
radio.set_radio_props({"facecolor": COR_DESTAQUE, "edgecolor": COR_SUAVE,
                       "linewidth": 1.5})


# funções de atualização
def parametros():
    """Lê os valores atuais dos sliders (Δt convertido para segundos)."""
    return (s_m1.val, s_m2.val, s_v1.val, s_v2.val, s_dt.val / 1000.0)


def tc_atual(m1, m2, v1, v2):
    """Instante do choque para os parâmetros atuais (None se não colide)."""
    meia = (largura(m1) + largura(m2)) / 2
    return instante_colisao(X1_0, X2_0, v1, v2, meia)


def empilhar(eixo, antes, depois, titulo, mostrar_total_linha):
    """
    Barras empilhadas por bloco (azul = bloco 1, vermelho = bloco 2).
    antes/depois = (valor do bloco 1, valor do bloco 2).
    Valores negativos são empilhados para baixo do zero.
    """
    eixo.clear()
    eixo.set_facecolor(COR_CARTAO)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color(COR_BORDA)
    eixo.tick_params(labelsize=8.5, length=3, color=COR_BORDA,
                     labelcolor=COR_SUAVE)
    eixo.grid(axis="y", color=COR_GRADE, lw=0.8)
    eixo.set_axisbelow(True)

    y_max, y_min = 0.0, 0.0
    for pos, (a, b) in enumerate((antes, depois)):
        topo_b, fundo_b = 0.0, 0.0
        for valor, cor, nome in ((a, COR1, "bloco 1"), (b, COR2, "bloco 2")):
            base_b = topo_b if valor >= 0 else fundo_b
            rotulo = nome if pos == 0 else None
            eixo.bar(pos, valor, bottom=base_b, color=cor, ec=COR_CARTAO,
                     lw=1.2, width=0.5, label=rotulo)
            if valor >= 0:
                topo_b += valor
            else:
                fundo_b += valor
        y_max, y_min = max(y_max, topo_b), min(y_min, fundo_b)
        total = a + b
        if mostrar_total_linha:  # linha tracejada = total (soma dos blocos)
            eixo.plot([pos - 0.33, pos + 0.33], [total, total], "--",
                      color=COR_TEXTO, lw=1.4)
        eixo.text(pos, max(topo_b, 0), f"{total:.2f}", ha="center",
                  va="bottom", fontsize=9.5, fontweight="bold", color=COR_TEXTO)
    eixo.axhline(0, color=COR_SUAVE, lw=0.8)
    eixo.set_xticks([0, 1])
    eixo.set_xticklabels(["antes da\ncolisão", "depois da\ncolisão"], fontsize=8.5)
    # título à esquerda e legenda à direita, na mesma faixa acima do gráfico
    eixo.set_title(titulo, loc="left", fontsize=10, fontweight="bold",
                   color=COR_TEXTO, pad=10)
    folga = 0.25 * ((y_max - y_min) or 1.0)   # espaço para o número
    eixo.set_ylim(y_min - (folga / 2 if y_min < 0 else 0), y_max + folga)
    eixo.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                fontsize=8, frameon=False, handlelength=1.0,
                columnspacing=1.2, borderpad=0.2, labelcolor=COR_TEXTO)


def mostrar_secoes(com_colisao):
    """Alterna entre as seções de resultados e a mensagem de 'sem colisão'."""
    for artista in secoes_res:
        artista.set_visible(com_colisao)
    txt_sem_t.set_visible(not com_colisao)
    txt_sem.set_visible(not com_colisao)


def selo(artista, ok, texto):
    """Selo verde (✓) ou vermelho (✗) ao lado do título de uma seção."""
    artista.set_text(("✓ " if ok else "✗ ") + texto)
    artista.set_color(COR_OK if ok else COR_ERRO)


def atualizar_painel(m1, m2, v1, v2, dt):
    """Calcula tudo e escreve o painel de resultados + gráficos."""
    if v1 <= v2:  # R5: caso sem colisão
        mostrar_secoes(False)
        txt_sem.set_text(
            f"v₁ = {v1:+.2f} m/s\n"
            f"v₂ = {v2:+.2f} m/s\n\n"
            "É necessário v₁ > v₂ para\n"
            "o bloco 1 alcançar o bloco 2."
        )
        empilhar(ax_p, (m1 * v1, m2 * v2), (m1 * v1, m2 * v2),
                 "Momento total p [kg·m/s]", True)
        empilhar(ax_k, (0.5 * m1 * v1**2, 0.5 * m2 * v2**2),
                 (0.5 * m1 * v1**2, 0.5 * m2 * v2**2),
                 "Energia cinética total K [J]", False)
        txt_aviso.set_text("Sem colisão: é preciso v1 > v2")
        return

    mostrar_secoes(True)
    u1, u2 = colisao_elastica(m1, m2, v1, v2)
    p_a, p_d = momento(m1, v1, m2, v2), momento(m1, u1, m2, u2)
    k_a, k_d = cinetica(m1, v1, m2, v2), cinetica(m1, u1, m2, u2)
    J1, J2, Fmed = impulsos(m1, m2, v1, v2, u1, u2, dt)
    tc = tc_atual(m1, m2, v1, v2)

    # verificações (tolerância numérica de ponto flutuante)
    ok_p = abs(p_d - p_a) < 1e-9
    ok_k = abs(k_d - k_a) < 1e-9
    dK = round(k_d - k_a, 9) + 0.0  # evita "-0.000"

    txt_cv.set_text(
        f"tc = {tc:.3f} s\n"
        f"v₁ = {v1:+7.3f}  →  u₁ = {u1:+7.3f}\n"
        f"v₂ = {v2:+7.3f}  →  u₂ = {u2:+7.3f}"
    )
    txt_p.set_text(f"antes  = {p_a:8.3f}\ndepois = {p_d:8.3f}")
    selo(selo_p, ok_p, "conservado")
    txt_k.set_text(f"antes  = {k_a:8.3f}\ndepois = {k_d:8.3f}\nΔK     = {dK:8.3f}")
    selo(selo_k, ok_k, "ΔK = 0")
    txt_j.set_text(
        f"J₁ = {J1:+8.3f} N·s\n"
        f"J₂ = {J2:+8.3f} N·s\n"
        f"F_med = {Fmed:7.1f} N (Δt = {dt*1000:.0f} ms)"
    )
    empilhar(ax_p, (m1 * v1, m2 * v2), (m1 * u1, m2 * u2),
             "Momento total p [kg·m/s]", True)
    empilhar(ax_k, (0.5 * m1 * v1**2, 0.5 * m2 * v2**2),
             (0.5 * m1 * u1**2, 0.5 * m2 * u2**2),
             "Energia cinética total K [J]", False)
    txt_aviso.set_text("")


def estado_cinematico(t, m1, m2, v1, v2):
    """Posições e velocidades dos blocos no instante t (MRU antes e depois de tc)."""
    tc = tc_atual(m1, m2, v1, v2)
    if tc is None or t < tc:
        return X1_0 + v1 * t, X2_0 + v2 * t, v1, v2
    u1, u2 = colisao_elastica(m1, m2, v1, v2)
    x1c = X1_0 + v1 * tc
    x2c = X2_0 + v2 * tc
    return x1c + u1 * (t - tc), x2c + u2 * (t - tc), u1, u2


def tempo_total(m1, m2, v1, v2):
    """Tempo até reiniciar: instante do choque + alguns segundos."""
    tc = tc_atual(m1, m2, v1, v2)
    return 6.0 if tc is None else tc + T_APOS_CHOQUE


def fmt_v(v):
    """Formata velocidade evitando '+0.0'."""
    return "0.0 m/s" if abs(v) < 0.05 else f"{v:+.1f} m/s"


def desenhar_seta(seta, x, y, v):
    """Seta horizontal de comprimento proporcional à velocidade."""
    if abs(v) < 0.05:
        seta.set_visible(False)
        return
    seta.set_visible(True)
    seta.xy = (x + ESC_SETA * v, y)   # ponta da seta
    seta.set_position((x, y))         # início da seta


def desenhar_blocos(t, atualizar_texto=True):
    m1, m2, v1, v2, _ = parametros()
    x1, x2, vv1, vv2 = estado_cinematico(t, m1, m2, v1, v2)
    w1, w2 = largura(m1), largura(m2)

    bloco1.set_xy((x1 - w1 / 2, 0)); bloco1.set_width(w1); bloco1.set_height(w1)
    bloco2.set_xy((x2 - w2 / 2, 0)); bloco2.set_width(w2); bloco2.set_height(w2)
    txt_b1.set_position((x1, w1 / 2)); txt_b1.set_text(f"1\n{m1:.1f} kg")
    txt_b2.set_position((x2, w2 / 2)); txt_b2.set_text(f"2\n{m2:.1f} kg")

    desenhar_seta(seta1, x1, w1 + 0.3, vv1)
    desenhar_seta(seta2, x2, w2 + 0.3, vv2)
    txt_v1.set_position((x1, w1 + 0.5)); txt_v1.set_text(fmt_v(vv1))
    txt_v2.set_position((x2, w2 + 0.5)); txt_v2.set_text(fmt_v(vv2))

    tc = tc_atual(m1, m2, v1, v2)

    # painel ao vivo: p(t) e K(t) no instante atual
    if tc is None:
        fase = "sem colisão"
    else:
        fase = "depois do choque" if t >= tc else "antes do choque"
    if atualizar_texto:   # texto mais pesado de desenhar: não precisa de todo quadro
        txt_status.set_text(
            f"t = {t:.2f} s  ({fase})\n"
            f"p(t) = {momento(m1, vv1, m2, vv2):.3f} kg·m/s\n"
            f"K(t) = {cinetica(m1, vv1, m2, vv2):.3f} J"
        )

    # marca do choque: aparece quando t >= tc
    if tc is not None and t >= tc:
        x_ch = X1_0 + v1 * tc + w1 / 2  # ponto de contato
        linha_choque.set_xdata([x_ch, x_ch])
        linha_choque.set_visible(True)
        txt_choque.set_position((x_ch + 0.2, Y_FAIXA))
        txt_choque.set_text(f"choque em t = {tc:.2f} s")
        txt_choque.set_visible(True)
    else:
        linha_choque.set_visible(False)
        txt_choque.set_visible(False)


def pintar_botao_pausa(pausado):
    """Botão azul 'Pausar' / verde 'Continuar'."""
    cor = P["continuar"] if pausado else COR_DESTAQUE
    hover = P["continuar_hover"] if pausado else P["destaque_hover"]
    b_pause.color, b_pause.hovercolor = cor, hover
    ax_pause.set_facecolor(cor)
    b_pause.label.set_text("Continuar" if pausado else "Pausar")


def reiniciar(_=None):
    """Reinicia a simulação com os valores atuais dos controles."""
    estado["t"] = 0.0
    estado["pausado"] = False
    pintar_botao_pausa(False)
    atualizar_painel(*parametros())
    desenhar_blocos(0.0)
    estado["redesenhar"] = True
    fig.canvas.draw_idle()


def alternar_pausa(_=None):
    estado["pausado"] = not estado["pausado"]
    pintar_botao_pausa(estado["pausado"])
    estado["redesenhar"] = True
    fig.canvas.draw_idle()


def ao_pressionar_tecla(evento):
    """Atalho: barra de espaço pausa/continua."""
    if evento.key in (" ", "space"):
        alternar_pausa()


def ao_mudar_slider(_):
    if estado["ignorar_eventos"]:
        return
    reiniciar()


def ao_escolher_caso(label):
    """Menu de casos: define m1, m2, v1 e v2 = 0."""
    m1, m2, v1 = CASOS[label]
    estado["ignorar_eventos"] = True
    s_m1.set_val(m1)
    s_m2.set_val(m2)
    s_v1.set_val(v1)
    s_v2.set_val(0.0)
    estado["ignorar_eventos"] = False
    reiniciar()


def quadro(_):
    """Chamada pelo FuncAnimation a cada frame (só atualiza os blocos)."""
    # tempo real decorrido desde o quadro anterior (limitado para não dar saltos)
    agora = time.perf_counter()
    passo = 0.0 if estado["ultimo"] is None else min(agora - estado["ultimo"], DT_MAX)
    estado["ultimo"] = agora
    if not estado["pausado"]:
        estado["t"] += passo * VEL_ANIM
        m1, m2, v1, v2, _dt = parametros()
        x1, x2, vv1, vv2 = estado_cinematico(estado["t"], m1, m2, v1, v2)
        w1, w2 = largura(m1), largura(m2)
        # reinicia ANTES de algum bloco, rótulo ou seta encostar na borda do eixo
        esquerda = min(x1 - w1 / 2, x1 - ROTULO_M, x1 + ESC_SETA * vv1,
                       x2 - w2 / 2, x2 + ESC_SETA * vv2)
        direita = max(x1 + w1 / 2, x1 + ESC_SETA * vv1,
                      x2 + w2 / 2, x2 + ROTULO_M, x2 + ESC_SETA * vv2)
        na_borda = esquerda < XMIN + 0.1 or direita > XMAX - 0.1
        if estado["t"] > tempo_total(m1, m2, v1, v2) or na_borda:
            estado["t"] = 0.0
            estado["ultimo"] = None
        estado["pronto"] = True
    estado["n"] += 1
    desenhar_blocos(estado["t"], atualizar_texto=(estado["n"] % 3 == 0))
    return (bloco1, bloco2, txt_b1, txt_b2, txt_v1, txt_v2,
            seta1, seta2, linha_choque, txt_choque, txt_status, txt_aviso)

def redesenhar_animados(_evento):
    """
    O redesenho completo (feito ao mexer num controle) pula os elementos
    animados; aqui um quadro da animação é desenhado logo em seguida
    """
    if estado["pronto"] and estado["redesenhar"]:
        estado["redesenhar"] = False
        estado["ultimo"] = None   # o quadro extra não avança o tempo
        anim._step()

# conexões e execução
for s in (s_m1, s_m2, s_v1, s_v2, s_dt):
    s.on_changed(ao_mudar_slider)
b_pause.on_clicked(alternar_pausa)
b_reset.on_clicked(reiniciar)
radio.on_clicked(ao_escolher_caso)
fig.canvas.mpl_connect("key_press_event", ao_pressionar_tecla)
fig.canvas.mpl_connect("draw_event", redesenhar_animados)

ao_escolher_caso("Caso 1: m1 = m2")   # estado inicial
anim = FuncAnimation(fig, quadro, interval=10, blit=True,
                     cache_frame_data=False)

plt.show()