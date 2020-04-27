package com.example.mya;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import static android.os.Build.VERSION_CODES.M;



public class MainActivity extends AppCompatActivity {
    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 3;
    private static final int IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255.0f;

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;

    private  String chosen = "converted_model.tflite";
    private Boolean quant = false;



    // selected classifier information received from extras


    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 224;
    private int DIM_IMG_SIZE_Y = 224;
    private int DIM_PIXEL_SIZE = 3;

    // int array to hold image data
    private int[] intValues;
    private ImageView imageView1,imageView2,imageView3,imageView4;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
        imageView1 = findViewById(R.id.doggy);
        imageView2 = findViewById(R.id.doggy2);
        imageView3 = findViewById(R.id.doggy4);
        imageView4 = findViewById(R.id.doggy3);

        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            Toast.makeText(getApplicationContext(),"STARTED",Toast.LENGTH_LONG).show();
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }
        if(quant){
            imgData =
                    ByteBuffer.allocateDirect(
                            DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        } else {
            imgData =
                    ByteBuffer.allocateDirect(
                            4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        }
        imgData.order(ByteOrder.nativeOrder());
        // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
        if(quant){
            labelProbArrayB= new byte[1][labelList.size()];
        } else {
            labelProbArray = new float[1][labelList.size()];
        }

        imageView1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap_orig = ((BitmapDrawable)imageView1.getDrawable()).getBitmap();
                //Bitmap bitmap_orig = BitmapFactory.decodeResource(getResources(),R.drawable.cat1);
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                if(quant){
                    tflite.run(imgData, labelProbArrayB);
                } else {
                    tflite.run(imgData, labelProbArray);
                    Float prob = labelProbArray[0][0];
                    Float prob2 = labelProbArray[0][1];
                    if(prob>prob2){
                        Toast.makeText(getApplicationContext(),"cat----->"+ String.valueOf(prob*100)+"%",Toast.LENGTH_SHORT).show();
                    }else {
                        Toast.makeText(getApplicationContext(),"dog----->"+ String.valueOf(prob2*100)+"%",Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });

        imageView2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap_orig = ((BitmapDrawable)imageView2.getDrawable()).getBitmap();
                //Bitmap bitmap_orig = BitmapFactory.decodeResource(getResources(),R.drawable.cat1);
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                if(quant){
                    tflite.run(imgData, labelProbArrayB);
                } else {
                    tflite.run(imgData, labelProbArray);
                    Float prob = labelProbArray[0][0];
                    Float prob2 = labelProbArray[0][1];
                    if(prob>prob2){
                        Toast.makeText(getApplicationContext(),"cat----->"+ String.valueOf(prob*100)+"%",Toast.LENGTH_SHORT).show();
                    }else {
                        Toast.makeText(getApplicationContext(),"dog----->"+ String.valueOf(prob2*100)+"%",Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });

        imageView3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap_orig = ((BitmapDrawable)imageView3.getDrawable()).getBitmap();
                //Bitmap bitmap_orig = BitmapFactory.decodeResource(getResources(),R.drawable.cat1);
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                if(quant){
                    tflite.run(imgData, labelProbArrayB);
                } else {
                    tflite.run(imgData, labelProbArray);
                    Float prob = labelProbArray[0][0];
                    Float prob2 = labelProbArray[0][1];
                    if(prob>prob2){
                        Toast.makeText(getApplicationContext(),"cat----->"+ String.valueOf(prob*100)+"%",Toast.LENGTH_SHORT).show();
                    }else {
                        Toast.makeText(getApplicationContext(),"dog----->"+ String.valueOf(prob2*100)+"%",Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });

        imageView4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap bitmap_orig = ((BitmapDrawable)imageView4.getDrawable()).getBitmap();
                //Bitmap bitmap_orig = BitmapFactory.decodeResource(getResources(),R.drawable.cat1);
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                if(quant){
                    tflite.run(imgData, labelProbArrayB);
                } else {
                    tflite.run(imgData, labelProbArray);
                    Float prob = labelProbArray[0][0];
                    Float prob2 = labelProbArray[0][1];
                    if(prob>prob2){
                        Toast.makeText(getApplicationContext(),"cat----->"+ String.valueOf(prob*100)+"%",Toast.LENGTH_SHORT).show();
                    }else {
                        Toast.makeText(getApplicationContext(),"dog----->"+ String.valueOf(prob2*100)+"%",Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });



    }
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(chosen);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                if(quant){
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }
    }
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }
}
