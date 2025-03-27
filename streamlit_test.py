import streamlit as st

# 设置页面标题
st.title('Streamlit示例')

# 创建一个滑块，范围从0到100，初始值为50
value = st.slider('请选择一个值', 0, 100, 50)

# 显示选择的值
st.write(f'你选择的值是: {value}')