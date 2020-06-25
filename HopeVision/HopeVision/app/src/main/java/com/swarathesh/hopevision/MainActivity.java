/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.swarathesh.hopevision;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.widget.ImageView;
import android.widget.TextView;

import android.speech.tts.TextToSpeech;
import android.widget.Toast;

import com.github.nisrulz.sensey.Sensey;
import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.text.Normalizer;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

//import senesy
import com.github.nisrulz.sensey.ProximityDetector;


public class MainActivity extends AppCompatActivity implements TextToSpeech.OnInitListener{
    //size of sampling
    private static final int Size = 224;
    //size of mean for sampling
    private static final int Mean = 117;
    private static final float std = 1;
    private static final String INPUT = "input";
    private static final String OUTPUT = "output";

    //implement text to speech for reading out the results to users
    private TextToSpeech tts;

    //import assets
    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private Classifier classifier;
    private Executor Theadexecutor = Executors.newSingleThreadExecutor();
    private TextView Result;
    private ImageView imageViewResult;
    private CameraView cameraView;
    private Sensey sensey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraView = (CameraView) findViewById(R.id.cameraView);
        imageViewResult = (ImageView) findViewById(R.id.imageViewResult);
        Result = (TextView) findViewById(R.id.textViewResult);

        Result.setMovementMethod(new ScrollingMovementMethod());

        //initiate sensey
        Sensey.getInstance().init(getApplicationContext());



        tts = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {
                if (i != TextToSpeech.ERROR){
                    tts.setLanguage(Locale.CANADA);
                }
            }
        });


        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bitmap = cameraKitImage.getBitmap();

                bitmap = Bitmap.createScaledBitmap(bitmap, Size, Size, false);

                imageViewResult.setImageBitmap(bitmap);

                final List<Classifier.Recognition> results = classifier.recognizeImage(bitmap);

                Result.setText(results.toString());

                String speak = results.toString();

                String input = speak;
                String withoutAccent = Normalizer.normalize(input, Normalizer.Form.NFD);
                String output = withoutAccent.replaceAll("[^a-zA-Z ]", "");


                tts.speak("i see "+output,TextToSpeech.QUEUE_FLUSH,null);



            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }
        });

        //initaite the proximity sensor
        ProximityDetector.ProximityListener proximityListener = new ProximityDetector.ProximityListener() {
            @Override
            public void onFar() {
                Toast.makeText(getApplicationContext(),"object to far make the user come closer !",Toast.LENGTH_LONG).show();
            }

            @Override
            public void onNear() {
                cameraView.captureImage();
            }
        };




        //initate sensey
        Sensey.getInstance().startProximityDetection(proximityListener);




        initTensorFlowAndLoadModel();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Theadexecutor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
        Sensey.getInstance().stop();
    }

    private void initTensorFlowAndLoadModel() {
        Theadexecutor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TFclassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            Size,
                            Mean,
                            std,
                            INPUT,
                            OUTPUT);
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {

    }

    @Override
    public void onInit(int i) {

    }
}
