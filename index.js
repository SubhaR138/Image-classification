const classifier = knnClassifier.create();
const webcamElement=document.getElementById('webcam');
let net;

async function app(){
    console.log('loading mobilenet');
    net=await mobilenet.load();
    console.log("successfully loaded");

    const webcam = await tf.data.webcam(webcamElement);
    const addExample = async classId => {
 
    const img = await webcam.capture();
    const activation = net.infer(img, 'conv_preds');
    classifier.addExample(activation, classId);
    img.dispose();
};
document.getElementById('class-a').addEventListener('click', () => addExample(0));
document.getElementById('class-b').addEventListener('click', () => addExample(1));
document.getElementById('class-c').addEventListener('click', () => addExample(2));
while (true) {
    if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
        const activation = net.infer(img, 'conv_preds');
        
        const result = await classifier.predictClass(activation);

        const classes = ['A', 'B', 'C'];
        document.getElementById('console').innerText = `
          prediction: ${classes[result.label]}\n
          probability: ${result.confidences[result.label]}
        `;
   
    // Dispose the tensor to release the memory.
    img.dispose();
    }

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame();
  }
}
app();  