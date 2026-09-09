from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Hahow 線上課程市場分析與類別預測', page_icon='📊', layout='wide')
BASE = Path(__file__).parent
DATA = BASE / 'data'
REQUIRED = [
 'hahow_courses_clean_full_latest.csv','hahow_courses_clean_ml_latest.csv','hahow_courses_multicategory_latest.csv',
 'model_results_latest.csv','classification_report_latest.csv','confusion_matrix_latest.csv',
 'error_cases_latest.csv','feature_importance_by_class_latest.csv']

@st.cache_data
def load_data():
    missing = [f for f in REQUIRED if not (DATA/f).exists()]
    if missing:
        raise FileNotFoundError('缺少正式資料檔：' + ', '.join(missing))
    return {Path(f).stem: pd.read_csv(DATA/f) for f in REQUIRED}

try:
    d=load_data()
except Exception as e:
    st.error(f'資料載入失敗：{e}')
    st.stop()

full=d['hahow_courses_clean_full_latest']; ml=d['hahow_courses_clean_ml_latest']; multi=d['hahow_courses_multicategory_latest']
models=d['model_results_latest']; report=d['classification_report_latest']; cm=d['confusion_matrix_latest']; errors=d['error_cases_latest']; fi=d['feature_importance_by_class_latest']
ZH={'art':'藝術','career-skills':'職場技能','design':'設計','finance-and-investment':'投資理財','handmade':'手作','humanities':'人文','language':'語言','lifestyle':'生活品味','marketing':'行銷','music':'音樂','photography':'攝影','programming':'程式'}

st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1450px}.stTabs [data-baseweb="tab"]{font-size:1rem}
.small-note{color:#666;font-size:.92rem}.section-question{font-size:1.05rem;font-weight:600;margin:.25rem 0 .75rem 0}
</style>''', unsafe_allow_html=True)

def num(s): return pd.to_numeric(s, errors='coerce')
def spearman_label(r):
    a=abs(r)
    return '極弱相關' if a<.2 else '弱相關' if a<.4 else '中度相關' if a<.6 else '強相關' if a<.8 else '很強相關'
def stats(s):
    s=num(s).dropna()
    return {'n':len(s),'median':s.median(),'q1':s.quantile(.25),'q3':s.quantile(.75),'p99':s.quantile(.99),'max':s.max()}
def fmt(v, digits=1): return f'{v:,.{digits}f}'
def model_name(row): return str(row.get('model_label',row.get('model','')))

def top_misclass(cm_df, n=8):
    pred_cols=list(cm_df.columns[1:]); rows=[]
    for i, r in cm_df.iterrows():
        true_raw=str(r.iloc[0]); true_zh=true_raw.split('｜')[-1]
        for col in pred_cols:
            pred_zh=str(col).split('｜')[-1]
            if pred_zh!=true_zh:
                rows.append({'真實類別':true_zh,'預測類別':pred_zh,'誤判數':int(r[col])})
    out=pd.DataFrame(rows); out=out[out['誤判數']>0].sort_values('誤判數',ascending=False).head(n).copy()
    out['方向']=out['真實類別']+' → '+out['預測類別']; return out

# Header
st.title('Hahow 線上課程市場分析與類別預測')
st.caption('Python 小組報告')
st.markdown('本專題透過 Hahow 線上課程資料，進行資料蒐集、資料清理、探索式資料分析與機器學習，觀察不同課程類別的市場特徵，並建立 12 類課程的文字分類模型。')
cols=st.columns(4)
for c,l,v in zip(cols,['完整課程資料','ML 單標籤資料','跨分類課程','課程類別'],[len(full),len(ml),len(multi),full['source_category_zh'].nunique()]): c.metric(l,f'{v:,}')
st.info(f'市場分析使用完整清理資料 {len(full):,} 筆；機器學習使用 {len(ml):,} 筆單標籤資料。另有 {len(multi):,} 筆跨分類課程保留於完整資料，但不加入目前的 12 類單標籤分類模型。')

tabs=st.tabs(['資料探索','資料品質','變數關係','模型評估','模型診斷','模型可解釋性','結論與限制'])

with tabs[0]:
    st.header('1. 資料探索')
    st.markdown('<div class="section-question">問題：Hahow 的 12 類課程分布是否平均？哪些類別較多、哪些較少？</div>',unsafe_allow_html=True)
    cnt=full['source_category_zh'].value_counts().rename_axis('類別').reset_index(name='課程數')
    maxr=cnt.iloc[cnt['課程數'].argmax()]; minr=cnt.iloc[cnt['課程數'].argmin()]
    c1,c2,c3,c4=st.columns(4); c1.metric('課程總數',f'{len(full):,}'); c2.metric('類別數',full['source_category_zh'].nunique()); c3.metric('最大類別',f"{maxr['類別']}｜{maxr['課程數']} 門"); c4.metric('最小類別',f"{minr['類別']}｜{minr['課程數']} 門")
    fig=px.bar(cnt.sort_values('課程數'),x='課程數',y='類別',orientation='h',text='課程數',title='12 類課程數量（完整清理資料）'); fig.update_layout(height=520); st.plotly_chart(fig,use_container_width=True)
    top3='、'.join(cnt.head(3)['類別']); low3='、'.join(cnt.tail(3)['類別'])
    st.info(f'**分析重點：** 12 類課程的樣本數並非完全平均。樣本較多的類別包含 **{top3}**；相對較少的包含 **{low3}**。類別不平衡可能使模型較容易學習樣本較多的類別，因此後續除了 Accuracy，也會搭配 **Macro F1 與各類別 Recall** 評估。')

    st.subheader('主要數值欄位分布')
    metric=st.radio('想看哪個市場特徵？',['價格','學生數','評分','課程時長'],horizontal=True)
    mp={'價格':('price','價格（TWD）'),'學生數':('student_count','學生數'),'評分':('rating_value','評分'),'課程時長':('duration_minutes','課程時長（分鐘）')}; col,label=mp[metric]
    s=num(full[col]); ss=stats(s); a,b,c,d=st.columns(4); a.metric('有效樣本',f"{ss['n']:,}"); b.metric('Median',fmt(ss['median'],1)); c.metric('P99',fmt(ss['p99'],1)); d.metric('Maximum',fmt(ss['max'],1))
    if metric=='價格':
        show=s[(s.notna())&(s<=ss['p99'])]
        note=f'主圖顯示至 P99（{ss["p99"]:,.0f} TWD），原始最大值仍保留為 {ss["max"]:,.0f} TWD。視覺化限制範圍不等於刪除資料。'
        analysis='價格存在高度離群值；在沒有外部證據確認資料錯誤的情況下，本專題保留原始值，並使用既有 price_outlier_flag 與 P99 協助閱讀主要價格區間。0 元價格也保留，不直接視為缺值。'
    elif metric=='學生數':
        show=s.dropna(); note='排除缺值；X 軸使用 log 尺度呈現。'; analysis='學生人數呈長尾分布，少數熱門課程學生數遠高於一般課程，因此採對數尺度協助觀察主要資料分布；這只改變圖的尺度，不改變原始學生數。'
    elif metric=='課程時長':
        show=s[(s.notna())&(s<=ss['p99'])]; note=f'排除缺值；主圖顯示至 P99（{ss["p99"]:,.1f} 分鐘），原始最大值仍保留於上方摘要。'; analysis='課程時長也存在長尾，因此主圖以 P99 協助閱讀主要區間；缺少時長的課程不自行推估。'
    else:
        show=s.dropna(); note='排除缺值；缺值不視為 0。'; analysis='評分分析只使用具有 rating_value 的課程；「沒有評分資料」不等於「評分為 0」，因此不做 0 值填補。'
    fig=px.histogram(x=show,nbins=40,labels={'x':label,'y':'課程數'},title=f'{metric}分布')
    if metric=='學生數': fig.update_xaxes(type='log',title='學生數（log scale）')
    st.plotly_chart(fig,use_container_width=True); st.caption(note); st.info('**分析重點：** '+analysis)

with tabs[1]:
    st.header('2. 資料品質如何影響後續分析')
    st.markdown('<div class="section-question">問題：缺值、0 值與極端值會如何影響我們對市場與模型的解讀？</div>',unsafe_allow_html=True)
    vars=['price','student_count','rating_value','duration_minutes']; zhcol={'price':'價格','student_count':'學生數','rating_value':'評分','duration_minutes':'課程時長'}
    rows=[]
    for x in vars:
        s=num(full[x]); rows.append([zhcol[x],int(s.isna().sum()),int((s==0).sum()),s.notna().sum(),s.quantile(.99),s.max()])
    qc=pd.DataFrame(rows,columns=['欄位','缺值','0 值','有效樣本','P99','最大值'])
    q=pd.DataFrame({'欄位':[zhcol[x] for x in vars],'完整資料缺值':[full[x].isna().sum() for x in vars],'ML資料缺值':[ml[x].isna().sum() for x in vars]}).melt('欄位',var_name='資料集',value_name='缺值數')
    st.plotly_chart(px.bar(q,x='欄位',y='缺值數',color='資料集',barmode='group',text='缺值數',title='主要欄位缺值數'),use_container_width=True)
    st.dataframe(qc,use_container_width=True,hide_index=True)
    price_flag=int(full['price_outlier_flag'].fillna(False).astype(bool).sum()) if 'price_outlier_flag' in full.columns else None
    st.info(f'''**分析重點：**
- `rating_value` 缺值時，評分分析只使用有效資料，不補成 0，因為「沒有評分」不等於「0 分」。
- `student_count` 缺值時，涉及學生數的圖與相關分析只使用有效資料；`duration_minutes` 缺值也不自行推估。
- `price` 的 0 元保留為實際數值；極端價格不任意刪除，而以 `price_outlier_flag` 標記{'（目前標記 '+str(price_flag)+' 筆）' if price_flag is not None else ''}，並在需要時以 P99 限制**視覺化尺度**。
- 完整資料與 ML 資料的用途不同：市場 EDA 使用 {len(full):,} 筆；ML 僅使用 {len(ml):,} 筆單標籤資料，{len(multi):,} 筆跨分類課程不混入模型。''')

with tabs[2]:
    st.header('3. 變數關係')
    choice=st.radio('想回答哪個問題？',['價格較高的課程是否通常擁有更多學生？','評分較高的課程是否通常擁有更多學生？'],horizontal=False)
    if choice.startswith('價格'):
        x='price'; xlabel='價格（TWD）'; raw=full[['title','source_category_zh',x,'student_count']].copy(); tmp=raw.dropna().copy(); p99=num(tmp[x]).quantile(.99); tmp=tmp[num(tmp[x])<=p99]; note=f'價格顯示至 P99 = {p99:,.0f} TWD；排除缺值；學生數使用 log 軸。相關係數依目前圖中有效資料計算。'
    else:
        x='rating_value'; xlabel='評分'; tmp=full[['title','source_category_zh',x,'student_count']].dropna().copy(); note='排除缺值；學生數使用 log 軸。'
    r=tmp[[x,'student_count']].corr(method='spearman').iloc[0,1]; degree=spearman_label(r); direction='正' if r>0 else '負' if r<0 else '無'
    a,b,c=st.columns(3); a.metric('有效樣本',f'{len(tmp):,}'); b.metric('Spearman',f'{r:.3f}'); c.metric('相關程度',f'{degree}（{direction}向）')
    fig=px.scatter(tmp,x=x,y='student_count',color='source_category_zh',hover_name='title',log_y=True,opacity=.65,labels={x:xlabel,'student_count':'學生數（log scale）','source_category_zh':'類別'},title=choice); st.plotly_chart(fig,use_container_width=True); st.caption(note)
    subject='價格與學生人數' if x=='price' else '評分與學生人數'
    st.info(f'**分析重點：** {subject}的 Spearman 相關係數為 **{r:.3f}**，依本專題分級屬於 **{degree}**。因此目前資料並未呈現強烈的單調關係；即使觀察到相關，也**不能推論**改變價格或評分會造成學生人數改變。')

with tabs[3]:
    st.header('4. 模型評估')
    st.markdown('<div class="section-question">問題：三個文字分類模型的整體表現如何？為什麼正式採用 Logistic Regression？</div>',unsafe_allow_html=True)
    mean=models[models['fold'].astype(str)=='mean'].copy()
    if mean.empty or not {'accuracy','f1_macro','f1_weighted'}.issubset(mean.columns): st.error('model_results_latest.csv 缺少 mean 或必要指標欄位。')
    else:
        long=mean.melt(['model','model_label'],value_vars=['accuracy','f1_macro','f1_weighted'],var_name='指標',value_name='分數'); long['指標']=long['指標'].map({'accuracy':'Accuracy','f1_macro':'Macro F1','f1_weighted':'Weighted F1'})
        st.plotly_chart(px.bar(long,x='model_label',y='分數',color='指標',barmode='group',text_auto='.3f',range_y=[0,1],title='三模型 4-fold 平均表現'),use_container_width=True)
        high=mean.loc[mean['f1_macro'].idxmax()]; lr=mean[mean['model'].eq('tfidf_lr')].iloc[0] if mean['model'].eq('tfidf_lr').any() else None
        c1,c2=st.columns(2); c1.metric('Macro F1 數值最高模型',f"{model_name(high)}",f"{high['f1_macro']:.4f}")
        if lr is not None: c2.metric('專題正式採用模型','Logistic Regression',f"Macro F1 {lr['f1_macro']:.4f}")
        if lr is not None:
            diff=float(high['f1_macro']-lr['f1_macro'])
            st.info(f'**模型決策解讀：** Linear SVM 與 Logistic Regression 的 Macro F1 差距只有 **{diff:.4f}**，整體分類表現幾乎相同。數值最高模型是 **Linear SVM**；專題正式採用 **Logistic Regression**，主要因正式結果將其列為主要模型，且 Logistic Regression 可提供類別機率，較有利於後續信心值、錯誤分析與應用整合。這裡不把極小的分數差距解讀為顯著優劣。')
    cls=report[report['class'].isin(ZH)].copy(); cls['類別']=cls['class'].map(ZH)
    st.subheader('各類別 Recall / F1')
    l2=cls.melt('類別',value_vars=['recall','f1-score'],var_name='指標',value_name='分數'); l2['指標']=l2['指標'].replace({'recall':'Recall','f1-score':'F1'})
    st.plotly_chart(px.bar(l2,x='類別',y='分數',color='指標',barmode='group',range_y=[0,1.05],title='正式採用模型：各類別 Recall / F1'),use_container_width=True)
    top=cls.nlargest(4,'f1-score'); bottom=cls.nsmallest(4,'f1-score'); hum=cls[cls['class'].eq('humanities')]
    st.info(f"**分析重點：** F1 表現較佳的類別為 **{'、'.join(top['類別'])}**；相對較弱的為 **{'、'.join(bottom['類別'])}**。" + (f"其中人文類 Recall 為 **{hum.iloc[0]['recall']:.2f}**，代表實際屬於人文類的課程中，模型約成功辨識 {hum.iloc[0]['recall']*100:.0f}%；問題主要反映在漏判，而不是單看整體 Accuracy 就能發現。" if not hum.empty else ''))

with tabs[4]:
    st.header('5. 模型診斷')
    st.markdown('<div class="section-question">問題：模型最容易把哪些類別搞混？這些錯誤是否有共同特徵？</div>',unsafe_allow_html=True)
    labels=[str(c).split('｜')[-1] for c in cm.columns[1:]]; z=cm.iloc[:,1:].to_numpy(); true=[str(x).split('｜')[-1] for x in cm.iloc[:,0]]
    fig=go.Figure(go.Heatmap(z=z,x=labels,y=true,text=z,texttemplate='%{text}',colorscale='Blues',hovertemplate='真實：%{y}<br>預測：%{x}<br>筆數：%{z}<extra></extra>')); fig.update_layout(title='Logistic Regression Confusion Matrix',xaxis_title='預測類別',yaxis_title='真實類別',height=650); st.plotly_chart(fig,use_container_width=True)
    topn=top_misclass(cm,8); st.subheader('最常見誤判方向 Top 8'); st.dataframe(topn[['真實類別','預測類別','誤判數']],use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(topn.sort_values('誤判數'),x='誤判數',y='方向',orientation='h',text='誤判數',title='非對角線誤判筆數 Top 8'),use_container_width=True)
    st.info('**分析重點：** 誤判並非完全隨機，而是集中在部分語意邊界相近的課程類別。這表示 Hahow 課程內容本身具有跨領域特性；單一標籤分類會迫使模型在相近主題之間做出唯一選擇。')
    st.subheader('代表性 Error Cases')
    st.metric('正式誤判案例總數',f'{len(errors):,}')
    # Representative: high-confidence cases + known cross-domain Python finance if present, de-duplicated.
    chosen=[]
    pyfin=errors[errors['title'].astype(str).str.contains('Python.*理財|理財.*Python',regex=True,na=False)]
    if not pyfin.empty: chosen.append(pyfin.iloc[0])
    for _,r0 in errors.sort_values('predicted_confidence',ascending=False).iterrows():
        if all(r0['course_id']!=r['course_id'] for r in chosen): chosen.append(r0)
        if len(chosen)>=5: break
    rep=pd.DataFrame(chosen)
    for _,r0 in rep.iterrows():
        st.markdown(f"**{r0['title']}**  \n真實：**{r0['true_category_zh']}** → 預測：**{r0['predicted_category_zh']}**｜信心值 **{r0['predicted_confidence']:.3f}**")
        snippet=str(r0.get('text_snippet','')).replace('\n',' ')[:180]
        st.caption('文字線索節錄：'+snippet+('…' if len(str(r0.get('text_snippet','')))>180 else ''))
    st.info('**怎麼看 Error Cases：** 代表案例常同時出現多個領域的文字線索，例如 Python 與投資理財同時出現在同一門課。這類錯誤不一定表示模型完全沒有學到內容，也可能反映課程本身具有跨領域語意。上方案例由正式 error_cases CSV 動態挑選，不改動模型預測。')

with tabs[5]:
    st.header('6. 模型可解釋性')
    st.markdown('<div class="section-question">問題：模型判斷這個類別時，看重哪些文字？</div>',unsafe_allow_html=True)
    selected=st.selectbox('選擇課程類別',list(ZH),format_func=lambda x:ZH[x]); n=st.slider('顯示 Top N',5,15,8)
    f=fi[(fi['class']==selected)&(fi['rank']<=n)].sort_values('coefficient')
    if f.empty: st.warning('此類別沒有可顯示的 feature importance。')
    else: st.plotly_chart(px.bar(f,x='coefficient',y='feature',orientation='h',text_auto='.2f',labels={'coefficient':'Coefficient','feature':'文字特徵'},title=f'{ZH[selected]}：Top {n} 正向文字特徵'),use_container_width=True)
    st.info('**閱讀方式：** 下圖呈現 Logistic Regression 在判斷某一課程類別時，較具有辨識力的文字特徵。係數越高，代表該文字越能提高模型判斷為該類別的傾向；這是模型從資料學到的**統計關聯**，不代表現實世界中的因果關係。')

with tabs[6]:
    st.header('7. 結論與限制')
    means=models[models['fold'].astype(str)=='mean']; lr=means[means['model'].eq('tfidf_lr')].iloc[0] if means['model'].eq('tfidf_lr').any() else None
    cls=report[report['class'].isin(ZH)].copy(); hum=cls[cls['class'].eq('humanities')]
    price_s=stats(full['price']); miss_rating=int(full['rating_value'].isna().sum()); maxcat=cnt.iloc[0]; mincat=cnt.iloc[-1]
    st.subheader('1. 市場資料發現')
    st.markdown(f'- 本資料涵蓋 **{full.source_category_zh.nunique()} 類、{len(full):,} 門課程**；類別分布不完全平均，最多為 **{maxcat["類別"]} {maxcat["課程數"]} 門**，最少為 **{mincat["類別"]} {mincat["課程數"]} 門**。\n- 價格、學生人數與課程時長具有長尾特性，因此部分圖表使用 P99 或 log scale 協助閱讀，但不刪除原始資料。')
    st.subheader('2. 資料品質發現')
    st.markdown(f'- 評分欄位有 **{miss_rating} 筆缺值**；缺值不補成 0。價格原始最大值為 **{price_s["max"]:,.0f} TWD**，在沒有外部證據下保留原值並透過既有 outlier flag / P99 協助分析。\n- 市場分析與 ML 採不同口徑：EDA {len(full):,} 筆；ML {len(ml):,} 筆單標籤；{len(multi):,} 筆跨分類不混入模型。')
    st.subheader('3. 機器學習發現')
    if lr is not None: st.markdown(f'- Linear SVM 與 Logistic Regression 整體表現非常接近；專題正式採用 **Logistic Regression**，其 4-fold 平均 Macro F1 為 **{lr["f1_macro"]:.3f}**。\n- 不同類別的辨識能力仍有差異，因此不能只看整體 Accuracy。')
    st.subheader('4. 模型限制與錯誤')
    htxt=f'人文類 Recall 為 **{hum.iloc[0]["recall"]:.2f}**；' if not hum.empty else ''
    st.markdown(f'- {htxt}模型誤判較集中於語意相近的類別，且正式 error cases 共 **{len(errors):,} 筆**。\n- 課程本身可能跨領域，加上類別樣本數不均，模型結果不應被解讀為絕對正確分類。Feature coefficient 也只代表模型內部統計權重。')
    st.subheader('5. 未來應用')
    st.success('清理後的課程資料與模型結果**可進一步延伸**至 LINE Bot，規劃課程搜尋、課程追蹤、價格異動通知，以及根據使用者追蹤課程建立內容式推薦功能。推薦功能若尚未完成，本網站不宣稱已正式上線。')
    with st.expander(f'補充：查看 {len(multi):,} 筆跨分類課程（不納入目前單標籤 ML）'):
        showcols=[c for c in ['title','source_categories_zh','category_count'] if c in multi.columns]; st.dataframe(multi[showcols],use_container_width=True,hide_index=True)

st.divider(); st.caption('本成果網站僅讀取專題正式 CSV；不需要 .env、API Key、Token、Channel Secret、LINE User ID 或密碼。')
