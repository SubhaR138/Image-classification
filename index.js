let net;
async function app(){
    console.log('loading mobilenet');
    net=await mobilenet.load();
    console.log("successfully loaded");

    const imgEl=document.getElementById('img');
    const result=await net.classify(imgEl);
    console.log(result);
    }
    app();