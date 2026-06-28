// Word cloud data
const wordData = [
    {text: "not", size: 1682},
    {text: "people", size: 1233},
    {text: "terrorist", size: 1020},
    {text: "like", size: 936},
    {text: "from", size: 906},
    {text: "just", size: 901},
    {text: "all", size: 886},
    {text: "about", size: 830},
    {text: "israel", size: 749},
    {text: "there", size: 742},
    {text: "more", size: 736},
    {text: "terrorists", size: 720},
    {text: "because", size: 712},
    {text: "dont", size: 674},
    {text: "terrorism", size: 581},
    {text: "one", size: 569},
    {text: "right", size: 565},
    {text: "hamas", size: 563},
    {text: "think", size: 554},
    {text: "war", size: 539}
];

// Color schemes
const colorSchemes = [
    ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
    ['#48bb78', '#38a169', '#68d391', '#9ae6b4', '#22543d', '#276749'],
    ['#ed8936', '#dd6b20', '#f6ad55', '#fbd38d', '#7b341e', '#9c4221']
];

let currentColorScheme = 0;
let currentLayout = 'spiral';

function generateWordCloud() {
    const container = document.getElementById('word-cloud');
    if (!container) {
        console.error("Word cloud container not found!");
        return;
    }
    
    container.innerHTML = '';
    
    const width = container.offsetWidth;
    const height = container.offsetHeight;
    
    console.log(`Container size: ${width}x${height}`);
    
    // Calculate font sizes based on word frequency
    const maxSize = Math.max(...wordData.map(w => w.size));
    const minSize = Math.min(...wordData.map(w => w.size));
    
    wordData.forEach((word, index) => {
        const wordElement = document.createElement('div');
        wordElement.className = 'cloud-word';
        wordElement.textContent = word.text;
        wordElement.title = `${word.text}: ${word.size} occurrences`;
        
        // Calculate font size based on frequency (scaled between 20px and 60px)
        const fontSize = 20 + (word.size - minSize) / (maxSize - minSize) * 40;
        wordElement.style.fontSize = `${fontSize}px`;
        
        // Assign color from current scheme
        const colorIndex = index % colorSchemes[currentColorScheme].length;
        wordElement.style.color = colorSchemes[currentColorScheme][colorIndex];
        
        // Position the word
        if (currentLayout === 'spiral') {
            positionWordSpiral(wordElement, width, height, index);
        } else {
            positionWordGrid(wordElement, width, height, index);
        }
        
        container.appendChild(wordElement);
    });
    
    console.log("Word cloud generated successfully");
}

function positionWordSpiral(element, width, height, index) {
    // Simple spiral positioning
    const angle = index * 0.2;
    const radius = Math.min(width, height) * 0.3 * (1 + index / wordData.length * 0.5);
    
    const x = width / 2 + Math.cos(angle) * radius;
    const y = height / 2 + Math.sin(angle) * radius;
    
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    element.style.transform = `translate(-50%, -50%)`;
}

function positionWordGrid(element, width, height, index) {
    // Grid-like positioning
    const cols = 5;
    const rows = Math.ceil(wordData.length / cols);
    
    const col = index % cols;
    const row = Math.floor(index / cols);
    
    const x = (col + 0.5) * (width / cols);
    const y = (row + 0.5) * (height / rows);
    
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
    element.style.transform = `translate(-50%, -50%)`;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded, initializing word cloud...");
    
    // Add event listeners for controls
    const colorBtn = document.getElementById('color-scheme-btn');
    const layoutBtn = document.getElementById('layout-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    if (colorBtn) {
        colorBtn.addEventListener('click', function() {
            currentColorScheme = (currentColorScheme + 1) % colorSchemes.length;
            generateWordCloud();
        });
    }
    
    if (layoutBtn) {
        layoutBtn.addEventListener('click', function() {
            currentLayout = currentLayout === 'spiral' ? 'grid' : 'spiral';
            generateWordCloud();
        });
    }
    
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            currentColorScheme = 0;
            currentLayout = 'spiral';
            generateWordCloud();
        });
    }
    
    // Generate initial word cloud
    generateWordCloud();
});

// Regenerate on window resize
window.addEventListener('resize', generateWordCloud);