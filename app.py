import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis Supervivencia Titanic", 
    page_icon="🚢", 
    layout="wide"
)

# Título principal
st.title("🚢 ANÁLISIS DE SUPERVIVENCIA - TITANIC")
st.markdown("---")

# Carga de datos
@st.cache_data
def load_data():
    # Verificar si el archivo existe
    if not os.path.exists("Titanic-Dataset.csv"):
        st.error("❌ El archivo 'Titanic-Dataset.csv' no se encuentra en el directorio actual")
        st.info("""
        **Estructura esperada del CSV:**
        - PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
        """)
        return None
    
    try:
        df = pd.read_csv("Titanic-Dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Cargar datos
df = load_data()

if df is not None:
    # Sidebar para filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtros interactivos
    sex_filter = st.sidebar.multiselect(
        "Filtrar por Sexo:",
        options=df['Sex'].unique(),
        default=df['Sex'].unique()
    )
    
    # Manejar valores nulos en Age
    age_min = int(df['Age'].min()) if not df['Age'].isna().all() else 0
    age_max = int(df['Age'].max()) if not df['Age'].isna().all() else 100
    
    age_range = st.sidebar.slider(
        "Rango de Edad:",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max)
    )
    
    # Aplicar filtros
    filtered_df = df[
        (df['Sex'].isin(sex_filter)) & 
        (df['Age'] >= age_range[0]) & 
        (df['Age'] <= age_range[1])
    ].copy()
    
    # Limpiar datos para análisis de edad
    age_analysis_df = filtered_df.dropna(subset=['Age']).copy()
    
    # 1. ANÁLISIS UNIVARIADO
    st.header("1. ANÁLISIS UNIVARIADO")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Variable: SEXO")
        sex_counts = filtered_df['Sex'].value_counts()
        sex_percentages = filtered_df['Sex'].value_counts(normalize=True) * 100
        
        st.metric("Hombres", f"{sex_counts.get('male', 0)} ({sex_percentages.get('male', 0):.1f}%)")
        st.metric("Mujeres", f"{sex_counts.get('female', 0)} ({sex_percentages.get('female', 0):.1f}%)")
    
    with col2:
        st.subheader("📈 Variable: EDAD")
        if not age_analysis_df.empty:
            st.metric("Media", f"{age_analysis_df['Age'].mean():.1f} años")
            st.metric("Mediana", f"{age_analysis_df['Age'].median():.1f} años")
            
            mode_result = age_analysis_df['Age'].mode()
            moda = mode_result.iloc[0] if not mode_result.empty else 'N/A'
            st.metric("Moda", f"{moda} años")
            
            st.metric("Desviación estándar", f"{age_analysis_df['Age'].std():.1f} años")
            st.metric("Rango", f"{age_analysis_df['Age'].min():.0f} - {age_analysis_df['Age'].max():.0f} años")
            
            Q1 = age_analysis_df['Age'].quantile(0.25)
            Q3 = age_analysis_df['Age'].quantile(0.75)
            st.metric("Rango intercuartílico", f"{Q1:.1f} - {Q3:.1f} años")
    
    with col3:
        st.subheader("🆘 Variable: SUPERVIVENCIA")
        survival_counts = filtered_df['Survived'].value_counts()
        survival_percentages = filtered_df['Survived'].value_counts(normalize=True) * 100
        
        survived_count = survival_counts.get(1, 0)
        not_survived_count = survival_counts.get(0, 0)
        survived_percent = survival_percentages.get(1, 0)
        not_survived_percent = survival_percentages.get(0, 0)
        
        st.metric("Sobrevivieron", f"{survived_count} ({survived_percent:.1f}%)")
        st.metric("No sobrevivieron", f"{not_survived_count} ({not_survived_percent:.1f}%)")
    
    st.markdown("---")
    
    # 2. ANÁLISIS BIVARIADO
    st.header("2. ANÁLISIS BIVARIADO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Sexo vs Supervivencia")
        
        # Crear tabla de contingencia
        sex_survival = pd.crosstab(filtered_df['Sex'], filtered_df['Survived'], 
                                  rownames=['Sexo'], 
                                  colnames=['Supervivencia'])
        sex_survival.columns = ['No sobrevivieron', 'Sobrevivieron']
        sex_survival['Total'] = sex_survival.sum(axis=1)
        sex_survival['Tasa Supervivencia'] = (sex_survival['Sobrevivieron'] / sex_survival['Total'] * 100).round(1)
        
        # Mostrar tabla
        st.dataframe(sex_survival.style.format({
            'Tasa Supervivencia': '{:.1f}%'
        }))
    
    with col2:
        st.subheader("📊 Edad vs Supervivencia")
        
        if not age_analysis_df.empty:
            survived_age = age_analysis_df[age_analysis_df['Survived'] == 1]['Age']
            not_survived_age = age_analysis_df[age_analysis_df['Survived'] == 0]['Age']
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Edad promedio sobrevivientes", f"{survived_age.mean():.1f} años")
            with col2_2:
                st.metric("Edad promedio no sobrevivientes", f"{not_survived_age.mean():.1f} años")
            with col2_3:
                diff = abs(survived_age.mean() - not_survived_age.mean())
                st.metric("Diferencia de medias", f"{diff:.1f} años")
            
            # Grupos de edad
            def age_group(age):
                if age <= 12:
                    return 'Niños (0-12)'
                elif age <= 17:
                    return 'Adolescentes (13-17)'
                elif age <= 59:
                    return 'Adultos (18-59)'
                else:
                    return 'Adultos mayores (60+)'
            
            age_analysis_df['Grupo Edad'] = age_analysis_df['Age'].apply(age_group)
            age_group_survival = age_analysis_df.groupby('Grupo Edad').agg({
                'Survived': ['count', 'sum']
            }).round(0)
            age_group_survival.columns = ['Total', 'Sobrevivieron']
            age_group_survival['Tasa Supervivencia'] = (age_group_survival['Sobrevivieron'] / age_group_survival['Total'] * 100).round(1)
            
            st.dataframe(age_group_survival.style.format({
                'Tasa Supervivencia': '{:.1f}%'
            }))
    
    st.markdown("---")
    
    # 3. GRÁFICOS ESENCIALES CON PLOTLY
    st.header("📊 LOS 4 GRÁFICOS ESENCIALES")
    
    # Gráfico 1: Diagrama de Barras - Supervivencia por Sexo
    st.subheader("Gráfico 1: Supervivencia por Sexo")
    
    # Preparar datos para el gráfico
    survival_by_sex = filtered_df.groupby(['Sex', 'Survived']).size().reset_index(name='count')
    survival_by_sex['Supervivencia'] = survival_by_sex['Survived'].map({0: 'No Sobrevivieron', 1: 'Sobrevivieron'})
    
    fig1 = px.bar(
        survival_by_sex, 
        x='Sex', 
        y='count', 
        color='Supervivencia',
        barmode='group',
        color_discrete_map={'Sobrevivieron': '#51cf66', 'No Sobrevivieron': '#ff6b6b'},
        title='Supervivencia por Sexo'
    )
    fig1.update_layout(
        xaxis_title='Sexo',
        yaxis_title='Cantidad de Pasajeros',
        showlegend=True
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Histograma - Distribución de Edades
    st.subheader("Gráfico 2: Distribución de Edades por Supervivencia")
    
    if not age_analysis_df.empty:
        fig2 = px.histogram(
            age_analysis_df,
            x='Age',
            color='Survived',
            nbins=20,
            opacity=0.7,
            color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
            title='Distribución de Edades por Supervivencia',
            labels={'Survived': 'Supervivencia', 'Age': 'Edad (años)'}
        )
        fig2.update_layout(
            xaxis_title='Edad (años)',
            yaxis_title='Frecuencia',
            legend_title='Supervivencia'
        )
        # Actualizar leyenda
        fig2.for_each_trace(lambda t: t.update(name='Sobrevivieron' if t.name == '1' else 'No Sobrevivieron'))
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Diagrama de Caja - Edad vs Supervivencia
    st.subheader("Gráfico 3: Distribución de Edad por Estado de Supervivencia")
    
    if not age_analysis_df.empty:
        age_analysis_df['Supervivencia_Label'] = age_analysis_df['Survived'].map({0: 'No Sobrevivieron', 1: 'Sobrevivieron'})
        
        fig3 = px.box(
            age_analysis_df,
            x='Supervivencia_Label',
            y='Age',
            color='Supervivencia_Label',
            color_discrete_map={'Sobrevivieron': '#51cf66', 'No Sobrevivieron': '#ff6b6b'},
            title='Distribución de Edad por Estado de Supervivencia'
        )
        fig3.update_layout(
            xaxis_title='Estado de Supervivencia',
            yaxis_title='Edad (años)',
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Gráfico 4: Diagrama de Barras - Tasa de Supervivencia por Grupo Etario
    st.subheader("Gráfico 4: Tasa de Supervivencia por Grupo Etario")
    
    if not age_analysis_df.empty:
        fig4 = px.bar(
            age_group_survival.reset_index(),
            x='Grupo Edad',
            y='Tasa Supervivencia',
            color='Grupo Edad',
            title='Tasa de Supervivencia por Grupo Etario',
            text='Tasa Supervivencia'
        )
        fig4.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig4.update_layout(
            xaxis_title='Grupo de Edad',
            yaxis_title='Tasa de Supervivencia (%)',
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # 4. INTERPRETACIÓN ESTADÍSTICA COMPLETA
    st.header("📋 INTERPRETACIÓN ESTADÍSTICA COMPLETA")
    
    # Cálculos para conclusiones
    if not filtered_df.empty and not age_analysis_df.empty:
        male_survival_rate = sex_survival.loc['male', 'Tasa Supervivencia'] if 'male' in sex_survival.index else 0
        female_survival_rate = sex_survival.loc['female', 'Tasa Supervivencia'] if 'female' in sex_survival.index else 0
        
        survival_ratio = female_survival_rate / male_survival_rate if male_survival_rate > 0 else float('inf')
        
        st.subheader("🎯 Conclusiones Principales:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("✅ **Supervivencia por Sexo:**")
            st.metric("Ratio Mujeres/Hombres", f"{survival_ratio:.1f}x")
            st.write(f"- Las mujeres tuvieron **{survival_ratio:.1f} veces** más probabilidad de sobrevivir")
            st.write(f"- Tasa de supervivencia femenina: **{female_survival_rate}%**")
            st.write(f"- Tasa de supervivencia masculina: **{male_survival_rate}%**")
            
            st.write("✅ **Política 'Mujeres y niños primero':**")
            if 'Niños (0-12)' in age_group_survival.index:
                children_survival = age_group_survival.loc['Niños (0-12)', 'Tasa Supervivencia']
                st.metric("Supervivencia Niños", f"{children_survival}%")
                st.write("- La política se aplicó **parcialmente**, priorizando mujeres y niños")
        
        with col2:
            st.write("✅ **Impacto de la Edad:**")
            age_diff = abs(survived_age.mean() - not_survived_age.mean())
            st.metric("Diferencia de edad", f"{age_diff:.1f} años")
            
            if age_diff < 5:
                st.write("- La edad **no fue un factor determinante** en la supervivencia general")
            else:
                st.write("- La edad **pudo influir** significativamente en la supervivencia")
            
            st.write("✅ **Patrones por Grupos Etarios:**")
            for group in age_group_survival.index:
                rate = age_group_survival.loc[group, 'Tasa Supervivencia']
                total = age_group_survival.loc[group, 'Total']
                st.write(f"- **{group}**: {rate}% ({int(age_group_survival.loc[group, 'Sobrevivieron'])}/{total})")
    
    st.markdown("---")
    
    # 5. FLUJO RECOMENDADO DE PRESENTACIÓN
    st.header("🎭 FLUJO RECOMENDADO DE PRESENTACIÓN")
    
    steps = [
        "🎯 **1. Empezar con Gráfico 1 (Sexo vs Supervivencia)** - Impacto visual inmediato sobre la diferencia más notable",
        "👶 **2. Continuar con Gráfico 4 (Grupos de edad)** - Contexto demográfico y priorización en rescate", 
        "📊 **3. Mostrar Gráfico 2 (Distribución edades)** - Análisis detallado de la distribución por edad",
        "📋 **4. Finalizar con Gráfico 3 (Resumen estadístico edades)** - Conclusiones sólidas con medidas de tendencia central"
    ]
    
    for step in steps:
        st.write(step)
    
    # Información adicional en sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ Información")
    st.sidebar.metric("Total de pasajeros", len(filtered_df))
    st.sidebar.metric("Pasajeros con edad registrada", len(age_analysis_df))
    
    # Mostrar datos brutos
    with st.expander("🔍 Ver datos brutos"):
        st.dataframe(filtered_df)
    
else:
    st.error("No se pudieron cargar los datos. Por favor, verifica que el archivo 'Titanic-Dataset.csv' esté en el directorio correcto.")
