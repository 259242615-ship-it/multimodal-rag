"""Streamlit前端:多模态问答+检索仪表盘"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="多模态文档问答",page_icon="📚")
st.title("📚 多模态文档问答系统")

tab_ask,tab_dash,tab_up = st.tabs(["💬 问答", "🔬 检索仪表盘","📁 入库"])

#=====Tab1:问答=====
with tab_ask:
    mode = st.radio("检索方式",["纯向量","混合检索"],horizontal=True)
    question = st.text_input("问点什么")
    if st.button("提问") and question:
        with st.spinner("检索 + 生成中..."):
            r = requests.post(f"{API}/ask",json={"question":question,"use_hybrid":mode=="混合检索"},timeout=60)
        if r.status_code != 200:
            st.error(f"请求失败 {r.status_code}:{r.text}")
        else:
            data = r.json()
            st.subheader("💬 回答")
            st.write(data["answer"])
            st.subheader("📎 引用来源")
            mode = data.get("retrieval_mode","vector")
            hint = "(RRF分数不可跨模式比较)" if mode == "hybrid" else ""
            st.caption(f"检索方式:{mode}{hint}")
            for s in data["sources"]:
                score_name = "精排分" if "rerank" in mode else ("RRF排名分" if mode == "hybrid" else "相似度")
                label = f"[{s['index']}] 第{s['page']}页 {s['type']} {score_name}{s['score']:.3f}"
                if s['type'] == "image" and s["image_url"]:
                    #图片经过后端的url加载
                    st.image(f"{API}{s['image_url']}",caption=label)
                else:
                    with st.expander(label):
                        st.write(s["content"])


#=====tap2:检索仪表盘====
with tab_dash:
    q = st.text_input("检索查询",key="dash_q")
    c1,c2 = st.columns(2)
    top_k = c1.slider("top_k",1,10,5)
    only = c2.selectbox("类型筛选",["全部","text","image"])

    if st.button("检索",key="dash_btn") and q:
        params = {"q":q,"top_k":top_k}
        if only != "全部":
            params["only_type"] = only

        r = requests.get(f"{API}/search",params=params,timeout=30)
        results = r.json()["results"]

        if not results:
            st.warning("无结果")
        else:
            n_img = sum(1 for x in results if x['type']=='image')
            m1,m2,m3 = st.columns(3)
            m1.metric("结果数",len(results))
            m2.metric("图片数",n_img)
            m3.metric("最高分",f"{results[0]['score']:.3f}")

            st.dataframe([{
                "排名":i,
                "类型":x["type"],
                "页码":x["page"],
                "相关度":round(x['score'],3),
                "内容摘要":x['content'][:60],
            } for i,x in enumerate(results,start=1)])

            st.bar_chart({f"#{i}{x['type']}":x["score"] for i,x in enumerate(results,start=1)})

#====tap3:上传入库====
with tab_up:
    st.caption("上传 .txt/.md ->切块->向量化->增量入库(容器重启会丢)")
    uploaded = st.file_uploader("选择文档",type=["txt","md"])

    if not uploaded:
        st.info("先选择一个.txt或.md文件")
    else:
        if st.button("入库",key="up_btn"):
            with st.spinner("切块+向量化中(每块调用一次Embedding(API))..."):
                r = requests.post(f"{API}/upload",
                                  files={"file":(uploaded.name,uploaded.getvalue(),uploaded.type or "text/plain")},
                                  timeout=120,)
                if r.status_code != 200:
                    try:
                        msg = r.json().get("detail")
                    except Exception:
                        msg = r.text
                    st.error(f"入库失败{r.status_code}:{msg}")
                else:
                    d = r.json()
                    st.success(f"✅ {d['filename']} 入库{d['chunks']}块 (库内共有{d['total']}条)")




