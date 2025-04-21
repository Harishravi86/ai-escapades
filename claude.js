// email-with-charts.js
// Create charts locally and send them in an email
const { createCanvas } = require('canvas');
const Chart = require('chart.js');
const fs = require('fs');
const nodemailer = require('nodemailer');
const path = require('path');

// Configuration
const EMAIL_FROM = 'harishravi86@gmail.com';
const EMAIL_TO = 'harishravi86@gmail.com';
const CITY = 'New York'; // Replace with your city

// Configure Chart.js for Node.js environment
Chart.defaults.global.responsive = false;
Chart.defaults.global.animation = false;

/**
 * Generate a chart image and save it to disk
 * @param {Object} config - Chart.js configuration object
 * @param {string} outputPath - Path to save the output image
 * @param {number} width - Width of the chart in pixels
 * @param {number} height - Height of the chart in pixels
 * @returns {string} Path to the generated image
 */
function generateChart(config, outputPath, width = 600, height = 300) {
  console.log(`Creating chart: ${outputPath}`);
  // Create a Canvas instance
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Create the chart
  new Chart(ctx, config);
  
  // Write to file
  const buffer = canvas.toBuffer('image/png');
  fs.writeFileSync(outputPath, buffer);
  
  console.log(`Chart saved: ${outputPath}`);
  return outputPath;
}

/**
 * Generate all weather charts
 * @param {Object} weatherData - Weather data object
 * @returns {Object} Paths to generated chart images
 */
function generateWeatherCharts(weatherData) {
  // Create output directory if it doesn't exist
  const outputDir = path.join(__dirname, 'charts');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }
  
  // Temperature chart
  const tempChartConfig = {
    type: 'line',
    data: {
      labels: weatherData.dates,
      datasets: [
        {
          label: 'Temperature (°C)',
          data: weatherData.temperatures,
          fill: false,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }
      ]
    },
    options: {
      title: {
        display: true,
        text: `${weatherData.city}, ${weatherData.country} - Temperature Forecast`,
        fontSize: 16
      },
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: false
          }
        }]
      }
    }
  };
  
  // Weather distribution pie chart
  const pieChartConfig = {
    type: 'pie',
    data: {
      labels: Object.keys(weatherData.weatherDistribution).map(key => 
        `${key} (${weatherData.weatherDistribution[key]}%)`),
      datasets: [{
        data: Object.values(weatherData.weatherDistribution),
        backgroundColor: [
          'rgb(255, 196, 0)',    // Bright gold
          'rgb(107, 185, 240)',  // Sky blue
          'rgb(39, 125, 214)',   // Royal blue
          'rgb(75, 101, 132)',   // Navy blue
          'rgb(158, 173, 186)'   // Steel blue
        ],
        borderColor: 'white',
        borderWidth: 2
      }]
    },
    options: {
      title: {
        display: true,
        text: `${weatherData.city} - Weather Condition Distribution`,
        fontSize: 16
      }
    }
  };
  
  // Weather distribution donut chart
  const donutChartConfig = {
    type: 'doughnut',
    data: {
      labels: Object.keys(weatherData.weatherDistribution).map(key => 
        `${key} (${weatherData.weatherDistribution[key]}%)`),
      datasets: [{
        data: Object.values(weatherData.weatherDistribution),
        backgroundColor: [
          'rgb(255, 196, 0)',    // Bright gold
          'rgb(107, 185, 240)',  // Sky blue
          'rgb(39, 125, 214)',   // Royal blue
          'rgb(75, 101, 132)',   // Navy blue
          'rgb(158, 173, 186)'   // Steel blue
        ],
        borderColor: 'white',
        borderWidth: 2
      }]
    },
    options: {
      title: {
        display: true,
        text: `${weatherData.city} - Weather Outlook`,
        fontSize: 16
      },
      cutoutPercentage: 65
    }
  };
  
  // Generate all three charts
  const tempChartPath = generateChart(tempChartConfig, path.join(outputDir, 'temperature-chart.png'));
  const pieChartPath = generateChart(pieChartConfig, path.join(outputDir, 'weather-pie-chart.png'));
  const donutChartPath = generateChart(donutChartConfig, path.join(outputDir, 'weather-donut-chart.png'));
  
  return {
    temperatureChart: tempChartPath,
    pieChart: pieChartPath,
    donutChart: donutChartPath
  };
}

/**
 * Send email with weather charts
 * @param {Object} chartPaths - Paths to chart images
 * @param {Object} weatherData - Weather data
 */
async function sendEmailWithCharts(chartPaths, weatherData) {
  // Create reusable transporter
  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: 'harishravi86@gmail.com',
      pass: ''
    }
  });
  
  // Create daily forecast table with more professional styling
  let forecastTable = `
    <table style="width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 20px; font-family: Arial, sans-serif;">
      <tr style="background-color: #4b6584;">
        <th style="padding: 12px; text-align: left; border: 1px solid #ddd; color: white;">Date</th>
        <th style="padding: 12px; text-align: left; border: 1px solid #ddd; color: white;">Temperature (°C)</th>
        <th style="padding: 12px; text-align: left; border: 1px solid #ddd; color: white;">Condition</th>
      </tr>
  `;
  
  // Weather condition to emoji mapping
  const conditionEmoji = {
    'Clear': '☀️',
    'Clouds': '☁️',
    'Rain': '🌧️',
    'Drizzle': '🌦️',
    'Thunderstorm': '⛈️',
    'Snow': '❄️',
    'Mist': '🌫️',
    'Fog': '🌫️',
    'Haze': '🌫️'
  };
  
  for (let i = 0; i < weatherData.dates.length; i++) {
    const emoji = conditionEmoji[weatherData.conditions[i]] || '🌡️';
    const rowColor = i % 2 === 0 ? '#f8f9fa' : '#ffffff';
    
    forecastTable += `
      <tr style="background-color: ${rowColor};">
        <td style="padding: 10px; text-align: left; border: 1px solid #ddd; font-weight: bold;">${weatherData.dates[i]}</td>
        <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">${weatherData.temperatures[i].toFixed(1)}°C</td>
        <td style="padding: 10px; text-align: left; border: 1px solid #ddd;">${emoji} ${weatherData.conditions[i]}</td>
      </tr>
    `;
  }
  
  forecastTable += `</table>`;
  
  // Email content with improved styling
  const mailOptions = {
    from: EMAIL_FROM,
    to: EMAIL_TO,
    subject: `Weather Forecast for ${CITY}`,
    html: `
      <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; color: #333;">
        <h1 style="color: #4b6584; text-align: center; border-bottom: 2px solid #eee; padding-bottom: 10px;">
          Weather Forecast for ${weatherData.city}, ${weatherData.country}
        </h1>
        
        <div style="margin: 30px 0;">
          <h2 style="color: #4b6584;">Temperature Trend</h2>
          <div style="text-align: center;">
            <img src="cid:temperature-chart" alt="Temperature Forecast Chart" style="max-width: 100%; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" />
          </div>
        </div>
        
        <div style="margin: 30px 0; display: flex; justify-content: space-between; flex-wrap: wrap;">
          <div style="flex: 1; min-width: 300px; margin: 10px;">
            <h2 style="color: #4b6584;">Weather Distribution (Pie)</h2>
            <div style="text-align: center;">
              <img src="cid:weather-pie-chart" alt="Weather Distribution Pie Chart" style="max-width: 100%; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" />
            </div>
          </div>
          
          <div style="flex: 1; min-width: 300px; margin: 10px;">
            <h2 style="color: #4b6584;">Weather Distribution (Donut)</h2>
            <div style="text-align: center;">
              <img src="cid:weather-donut-chart" alt="Weather Distribution Donut Chart" style="max-width: 100%; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);" />
            </div>
          </div>
        </div>
        
        <div style="margin: 30px 0;">
          <h2 style="color: #4b6584;">Daily Forecast</h2>
          ${forecastTable}
        </div>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #777; font-size: 12px; text-align: center;">
          <p>This forecast was generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}.</p>
          <p>Data provided by OpenWeatherMap</p>
        </div>
      </div>
    `,
    attachments: [
      {
        filename: 'temperature-chart.png',
        path: chartPaths.temperatureChart,
        cid: 'temperature-chart'
      },
      {
        filename: 'weather-pie-chart.png',
        path: chartPaths.pieChart,
        cid: 'weather-pie-chart'
      },
      {
        filename: 'weather-donut-chart.png',
        path: chartPaths.donutChart,
        cid: 'weather-donut-chart'
      }
    ]
  };
  
  try {
    console.log('Sending email...');
    const info = await transporter.sendMail(mailOptions);
    console.log('Email sent:', info.response);
    return info;
  } catch (error) {
    console.error('Error sending email:', error);
    throw error;
  }
}

// Example weather data (replace with your actual API call)
const sampleWeatherData = {
  city: 'New York',
  country: 'US',
  dates: ['Apr 21', 'Apr 22', 'Apr 23', 'Apr 24', 'Apr 25'],
  temperatures: [22, 24, 19, 21, 25],
  conditions: ['Clear', 'Clouds', 'Rain', 'Clouds', 'Clear'],
  weatherDistribution: {
    'Clear': 38,
    'Light Rain': 30,
    'Rain': 3,
    'Cloudy': 18,
    'Partly Cloudy': 11
  }
};

// Main function
async function main() {
  try {
    console.log('Generating weather charts...');
    // In a real application, you would fetch this data from an API
    const chartPaths = generateWeatherCharts(sampleWeatherData);
    
    console.log('Sending email with charts...');
    await sendEmailWithCharts(chartPaths, sampleWeatherData);
    
    console.log('Process completed successfully!');
  } catch (error) {
    console.error('Error in main process:', error);
  }
}

// Run the main function
main();
