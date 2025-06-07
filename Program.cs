using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using PC4.Data;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection") ?? 
    throw new InvalidOperationException("Connection string 'DefaultConnection' not found.");

builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlite(connectionString));

builder.Services.AddDatabaseDeveloperPageExceptionFilter();

builder.Services.AddDefaultIdentity<IdentityUser>(options => options.SignIn.RequireConfirmedAccount = true)
    .AddEntityFrameworkStores<ApplicationDbContext>();

builder.Services.AddControllersWithViews();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseMigrationsEndPoint();
}
else
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");
app.MapRazorPages();


// ML.NET - ANÁLISIS DE SENTIMIENTOS
var mlContext = new MLContext();

// Cargar datos de entrenamiento de sentimientos desde carpeta "data"
var sentimentData = mlContext.Data.LoadFromTextFile<SentimentData>(
    "data/sentiment-data.tsv", hasHeader: true, separatorChar: '\t');

// Construir pipeline de clasificación binaria
var sentimentPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

// Entrenar modelo de sentimiento
var sentimentModel = sentimentPipeline.Fit(sentimentData);

// Guardar modelo
mlContext.Model.Save(sentimentModel, sentimentData.Schema, "sentiment_model.zip");

// Crear motor de predicción
var sentimentPredictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(sentimentModel);

// Prueba
var input = new SentimentData { Text = "Muy mala experiencia, no me gustó." };
var sentimentResult = sentimentPredictionEngine.Predict(input);
Console.WriteLine($"Predicción: {(sentimentResult.Prediction ? "Positivo" : "Negativo")} - Score: {sentimentResult.Probability:P2}");


// ML.NET - SISTEMA DE RECOMENDACIÓN
var ratingData = mlContext.Data.LoadFromTextFile<RatingData>(
    "data/ratings-data.csv", hasHeader: true, separatorChar: ',');

// Construir pipeline de recomendación
var recommendationPipeline = mlContext.Transforms.Conversion.MapValueToKey("userIdEncoded", nameof(RatingData.UserId))
    .Append(mlContext.Transforms.Conversion.MapValueToKey("productIdEncoded", nameof(RatingData.ProductId)))
    .Append(mlContext.Recommendation().Trainers.MatrixFactorization(new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "productIdEncoded",
        LabelColumnName = nameof(RatingData.Label),
        NumberOfIterations = 20,
        ApproximationRank = 100
    }));

// Entrenar modelo
var recommendationModel = recommendationPipeline.Fit(ratingData);

// Guardar modelo
mlContext.Model.Save(recommendationModel, ratingData.Schema, "recommendation_model.zip");

// Crear motor de predicción
var recommendationPredictionEngine = mlContext.Model.CreatePredictionEngine<RatingData, RatingPrediction>(recommendationModel);

// Simular recomendaciones
var allProducts = new[] { "P1", "P2", "P3", "P4", "P5" };
var userId = "U2";

var predictions = allProducts.Select(pid => new
{
    ProductId = pid,
    Score = recommendationPredictionEngine.Predict(new RatingData { UserId = userId, ProductId = pid }).Score
})
.OrderByDescending(p => p.Score)
.Take(3);

foreach (var p in predictions)
    Console.WriteLine($"Producto recomendado: {p.ProductId}, Score: {p.Score}");

app.Run();
