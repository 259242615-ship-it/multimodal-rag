"""Streamlit前端:多模态问答+检索仪表盘"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import requests
import streamlit as st

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="多模态文档问答",page_icon="📚")
st.title("📚 多模态文档问答系统")

tab_ask,tab_dash = st.tabs(["💬 问答", "🔬 检索仪表盘"])

#=====Tab1:问答=====
with tab_ask:
    question = st.text_input("问点什么")
    if st.button("提问") and question:
        with st.spinner("检索 + 生成中..."):
            r = requests.post(f"{API}/ask",json={"question":question},timeout=60)
        if r.status_code != 200:
            st.error(f"请求失败 {r.status_code}:{r.text}")
        else:
            data = r.json()
            st.subheader("💬 回答")
            st.write(data["answer"])
            st.subheader("📎 引用来源")
            for s in data["sources"]:
                label = f"[{s['index']}] 第{s['page']}页 {s['type']} 相似度{s['score']:.3f}"
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
            m1.metric("图片数",n_img)
            m1.metric("最高分",f"{results[0]['score']:.3f}")

            st.dataframe([{
                "排名":i,
                "类型":x["type"],
                "页码":x["page"],
                "相关度":round(x['score'],3),
                "内容摘要":x['content'][:60],
            } for i,x in enumerate(results,start=1)])

            st.bar_chart({f"#{i}{x['type']}":x["score"] for i,x in enumerate(results,start=1)})



