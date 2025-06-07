using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using PC4.Data;
using Microsoft.ML.Data;
using Microsoft.ML;
var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection") ?? throw new InvalidOperationException("Connection string 'DefaultConnection' not found.");
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
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
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

var mlContext = new MLContext();

var data = mlContext.Data.LoadFromTextFile<SentimentData>(
    "sentiment-data.tsv", hasHeader: true, separatorChar: '\t');

var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());

var model = pipeline.Fit(data);

// Guardar modelo
mlContext.Model.Save(model, data.Schema, "sentiment_model.zip");

var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

var input = new SentimentData { Text = "Muy mala experiencia, no me gustó." };
var result = predictionEngine.Predict(input);

// Resultado: Positivo o Negativo + Score
Console.WriteLine($"Predicción: {(result.Prediction ? "Positivo" : "Negativo")} - Score: {result.Probability:P2}");


app.Run();
