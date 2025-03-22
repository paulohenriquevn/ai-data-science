class EDAPipeline:
    def __init__(
        self,
        missing_analyzer,
        distribution_analyzer,
        correlation_analyzer,
        significance_analyzer,
        outlier_analyzer,
        feature_plan_generator,
        feature_engineer,
        normalization_plan_generator,
        normalizer,
        pca_plan_generator,
        pca_executor,
        feature_scorer=None
    ):
        self.missing_analyzer = missing_analyzer
        self.distribution_analyzer = distribution_analyzer
        self.correlation_analyzer = correlation_analyzer
        self.significance_analyzer = significance_analyzer
        self.outlier_analyzer = outlier_analyzer
        self.feature_plan_generator = feature_plan_generator
        self.feature_engineer = feature_engineer
        self.normalization_plan_generator = normalization_plan_generator
        self.normalizer = normalizer
        self.pca_plan_generator = pca_plan_generator
        self.pca_executor = pca_executor
        self.feature_scorer = feature_scorer

        self.reports = {}
        self.df_transformed = None
        self.df_pca = None

    def run(self, df):
        # Etapa 1: Análises iniciais
        self.reports['missing'] = self.missing_analyzer.analyze(df)
        self.reports['distribution'] = self.distribution_analyzer.analyze(df)
        self.reports['correlation'] = self.correlation_analyzer.analyze(df)
        self.reports['significance'] = self.significance_analyzer.analyze(df)
        self.reports['outliers'] = self.outlier_analyzer.analyze(df)

        # Etapa 2: Geração e aplicação de plano de features
        feature_plan = self.feature_plan_generator(
            distribution_report=self.reports['distribution'],
            outlier_report=self.reports['outliers'],
            significance_report=self.reports['significance'],
            correlation_report=self.reports['correlation']
        ).generate()
        df = self.feature_engineer(plan=feature_plan).transform(df)

        # Etapa 3: Normalização
        normalization_plan = self.normalization_plan_generator(
            distribution_report=self.reports['distribution'],
            outlier_report=self.reports['outliers']
        ).generate()
        self.normalizer = self.normalizer(normalization_plan)
        self.normalizer.fit(df)
        df_normalized = self.normalizer.transform(df)

        # Etapa 4: PCA
        pca_plan = self.pca_plan_generator(
            df=df_normalized,
            significance_report=self.reports['significance'],
            correlation_report=self.reports['correlation']
        ).generate()
        self.df_pca = self.pca_executor(pca_plan).fit_transform(df_normalized)

        # Etapa 5: Feature Scoring (opcional)
        if self.feature_scorer:
            self.reports['feature_score'] = self.feature_scorer.analyze(df, list(self.reports.values()))

        self.df_transformed = df_normalized
        return self
