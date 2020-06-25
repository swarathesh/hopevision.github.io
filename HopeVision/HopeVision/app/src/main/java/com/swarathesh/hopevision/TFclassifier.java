

package com.swarathesh.hopevision;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;


public class TFclassifier implements Classifier {

    private static final String TAG = "ImageClassifier";


    private static final int MAX_RES = 3;
    private static final float TH = 0.1f;

    // Config values.
    private String input;
    private String output;
    private int Size;
    private int Mean;
    private float Std;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] Values;
    private float[] floatVal;
    private float[] outputs;
    private String[] outNames;

    private TensorFlowInferenceInterface Interface;

    private boolean Stats = false;

    private TFclassifier() {
    }


    public static Classifier create(
            AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            int imageMean,
            float imageStd,
            String inputName,
            String outputName)
            throws IOException {
        TFclassifier c = new TFclassifier();
        c.input = inputName;
        c.output = outputName;


        String actualname = labelFilename.split("file:///android_asset/")[1];

        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualname)));
        String line;
        while ((line = br.readLine()) != null) {
            c.labels.add(line);
        }
        br.close();

        c.Interface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        int numClasses =
                (int) c.Interface.graph().operation(outputName).output(0).shape().size(1);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);


        c.Size = inputSize;
        c.Mean = imageMean;
        c.Std = imageStd;

        c.outNames = new String[]{outputName};
        c.Values = new int[inputSize * inputSize];
        c.floatVal = new float[inputSize * inputSize * 3];
        c.outputs = new float[numClasses];

        return c;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {

        bitmap.getPixels(Values, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < Values.length; ++i) {
            final int val = Values[i];
            floatVal[i * 3 + 0] = (((val >> 16) & 0xFF) - Mean) / Std;
            floatVal[i * 3 + 1] = (((val >> 8) & 0xFF) - Mean) / Std;
            floatVal[i * 3 + 2] = ((val & 0xFF) - Mean) / Std;
        }


        // Copy the input data into TensorFlow.

        Interface.feed(
                input, floatVal, new long[]{1, Size, Size, 3});

        // Run the inference call.

        Interface.run(outNames, Stats);


        // Copy the output Tensor back into the output array.

        Interface.fetch(output, outputs);



        PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        3,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {

                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });
        for (int i = 0; i < outputs.length; ++i) {
            if (outputs[i] > TH) {
                pq.add(
                        new Recognition(
                                "" + i, labels.size() > i ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RES);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        TraceCompat.endSection(); // "recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {
        Stats = debug;
    }

    @Override
    public String getStatString() {
        return Interface.getStatString();
    }

    @Override
    public void close() {
        Interface.close();
    }
}
