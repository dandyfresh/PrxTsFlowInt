/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package srctsflow;

import com.src.opencv.core.Constantes;
import com.src.opencv.dto.Annotation;
import com.src.opencv.dto.Bndbox;
import com.src.opencv.dto.Objeto;
import com.src.opencv.dto.StadsBasic;
import com.src.opencv.md.dto.item;
import com.src.opencv.md.intf.ModModel;
import com.src.opencv.md.intf.ModeloGrafico;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import org.opencv.core.Mat;
import org.opencv.dnn.Net;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

import org.tensorflow.types.UInt8;

/**
 *
 * @author cursi
 */
public class SRCTSFlow extends ModeloGrafico {

    public void Init(String path) {
        File f = new File(path);
        if (!f.exists()) {
            //          System.out.println("NO EXISTE LIBRERIA TSFLOW:" + path);
        }
        try {

            System.load(path);
            System.setProperty("jna.library.path", "32".equals(System.getProperty("sun.arch.data.model")) ? "lib/win32-x86" : "lib/win32-x86-64");
        } catch (Exception e) {
            //          System.out.println(e.getMessage());
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        try {
            SRCTSFlow mod = new SRCTSFlow();
            String ruta = "D:\\datos\\recme\\data\\tensor\\modelos_TF\\coco_resnet101\\faster_rcnn_resnet101_coco_2018_01_28\\saved_model";
            mod.Init("D:/java/librerias/TensorFlow/tensorflow_jni.dll");
            SavedModelBundle model = SavedModelBundle.load(ruta, "serve");

//            printSignature(model);
            Path dir = Paths.get("C:\\Users\\dandyfresh\\Videos\\pru3\\");
            List<String> imgs = Files.list(dir)
                    // Filtrar por archivos con extensión ".jpg" o ".jpeg"
                    .filter(path -> Files.isRegularFile(path)
                    && path.toString().toLowerCase().endsWith(".jpg")
                    || path.toString().toLowerCase().endsWith(".jpeg"))
                    // Convertir los paths a cadenas y colectar los resultados en una lista
                    .map(Path::toString)
                    .collect(Collectors.toList());

            //imgs.add("C:\\Users\\dandyfresh\\Videos\\pru3\\V_C_1713385653106_F_27280.JPEG");
            for (String filename : imgs) {
                // final String filename = args[0];
                long ini=System.currentTimeMillis();
                int[] imageSize = mod.getImageSize(filename);
                // //          System.out.println("width: " + imageSize[0] + " height: " + imageSize[1]);
                List<Tensor<?>> outputs = null;

                //    String res= new String(model.session().runner, StandardCharsets.UTF_8);
                try (Tensor<UInt8> input = mod.makeImageTensor(filename)) {
                    outputs
                            = model
                                    .session()
                            .runner()
                            .feed("image_tensor", input)
                            .fetch("detection_scores")
                            .fetch("detection_classes")
                            .fetch("detection_boxes")
                            .run();
                }
                try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);) {
                    Tensor<Float> classesT = outputs.get(1).expect(Float.class);
                    Tensor<Float> boxesT = outputs.get(2).expect(Float.class);
                    // All these tensors have:
                    // - 1 as the first dimension
                    // - maxObjects as the second dimension
                    // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
                    // This can be verified by looking at scoresT.shape() etc.

                    int maxObjects = (int) scoresT.shape()[1];
                    //          System.out.println(maxObjects);
                    float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
                    float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
                    float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
                    // Print all objects whose score is at least 0.5.
                    //          System.out.printf("* %s\n", filename);
                    boolean foundSomething = false;
                    for (int i = 0; i < scores.length; ++i) {
                        if (scores[i] > 0.01) {

                            foundSomething = true;
                            // //          System.out.printf("\tFound %-20s (score: %.4f)\n", (int) classes[i], scores[i]);
//                            System.out.printf("\tFound %d \t (score: %.4f) \t (xmin: %d \t ymin: %d \t xmax: %d \t ymax: %d)\n",
//                                    (int) classes[i], mod.update(scores[i]), (int) (boxes[i][1] * imageSize[0]),
//                                    (int) (boxes[i][0] * imageSize[1]), (int) (boxes[i][3] * imageSize[0]),
//                                    (int) (boxes[i][2] * imageSize[1]));
                        }
                    }
                    if (!foundSomething) {
                        //          System.out.println("No objects detected with a high enough score.");
                    }
                }
                System.out.println("tiempo:"+(System.currentTimeMillis()-ini));
            }
        } catch (Exception ex) {
            Logger.getLogger("").log(Level.SEVERE, null, ex);
        }
    }

    @Override
    public Annotation detectObject(String img, int modelIndex) {
        Annotation an = null;
        try {

            //          System.out.println(an);
        } catch (Exception ex) {
            Logger.getLogger(SRCTSFlow.class.getName()).log(Level.SEVERE, null, ex);
        }
        return an;
    }

    private Tensor<UInt8> makeImageTensor(String filename) throws IOException {
        // TODO Auto-generated method stub
        BufferedImage img = ImageIO.read(new File(filename));
        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            throw new IOException(
                    String.format(
                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                            img.getType(), filename));
        }
        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[]{BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));

    }

    private Tensor<UInt8> makeImageTensor(byte[] data, int alto, int ancho) throws IOException {
        // TODO Auto-generated method stub

        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
        bgr2rgb(data);
        final long BATCH_SIZE = 1;
        final long CHANNELS = 3;
        long[] shape = new long[]{BATCH_SIZE, alto, ancho, CHANNELS};
        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));

    }

    private void bgr2rgb(byte[] data) {
        // TODO Auto-generated method stub
        for (int i = 0; i < data.length; i += 3) {
            byte tmp = data[i];
            data[i] = data[i + 2];
            data[i + 2] = tmp;
        }
    }

    @Override
    public List<Annotation> detectObjects(ModModel model, List<String> imgs, HashMap<String, StadsBasic> datos, int tipo) {
        List<Annotation> res = new ArrayList();

        Session s = ((SavedModelBundle) model.getWeightModel()).session();
        //final String filename = "/home/jcgarciaca/data/java_tf/source/DetectObjects/images/galletas/image1.JPEG";

        for (String im : imgs) {
            try {
                Annotation an = new Annotation();
                final String filename = im;
                BufferedImage img = ImageIO.read(new File(filename));
                if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
                    throw new IOException(
                            String.format(
                                    "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
                                    img.getType(), filename));
                }
                Runner r = s.runner();

                byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
                // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
                bgr2rgb(data);
                final long BATCH_SIZE = 1;
                final long CHANNELS = 3;
                long[] shape = new long[]{BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
                Tensor<UInt8> input = Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
                int[] imageSize = {img.getWidth(), img.getHeight()};
                an.setAlto(img.getHeight());
                an.setAncho(img.getWidth());
                data = null;
                // //          System.out.println("width: " + imageSize[0] + " height: " + imageSize[1]);
                img.flush();
                List<Tensor<?>> outputs = null;

                outputs
                        = r.feed("image_tensor", input)
                                .fetch("detection_scores")
                                .fetch("detection_classes")
                                .fetch("detection_boxes")
                                .run();

                Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
                Tensor<Float> classesT = outputs.get(1).expect(Float.class);
                Tensor<Float> boxesT = outputs.get(2).expect(Float.class);

                int maxObjects = (int) scoresT.shape()[1];
                //          System.out.println(maxObjects);
                float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
                float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
                float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
                // Print all objects whose score is at least 0.5.
                //          System.out.printf("* %s\n", filename);
                scoresT.close();
                input.close();
                classesT.close();
                boxesT.close();
                boolean foundSomething = false;
                for (int i = 0; i < scores.length; ++i) {

                    double score = update(scores[i]);
                    if (score >= model.getMinScore()) {

                        foundSomething = true;
                        String cn = model.getItemName((int) classes[i]);
                        int ind = an.getIndxObj(cn);
                        StadsBasic sb = datos.get(cn);

                        if (ind < 0) {
                            Objeto ob = new Objeto();
                            ob.setName(cn);
                            if (sb != null) {
                                ob.setLabel(sb.getLabel());
                            }
                            ob.setTipo(tipo);
                            an.getObject().add(ob);
                            ind = an.getObject().size() - 1;
                        } else {
                            if (sb != null) {
                                an.getObject().get(ind).setLabel(sb.getLabel());
                            }
                        }
                        Bndbox bn = new Bndbox();

                        bn.setScore(score);
                        bn.setId(cn);
                        bn.setDesc("T:" + cn);
                        bn.setXmin((int) (boxes[i][1] * imageSize[0]));
                        bn.setYmin((int) (boxes[i][0] * imageSize[1]));
                        bn.setXmax((int) (boxes[i][3] * imageSize[0]));
                        bn.setYmax((int) (boxes[i][2] * imageSize[1]));
                        bn.setTipo(tipo);
                        double prop = 1;
                        if (sb != null) {
                            if (bn.getOrient() == Constantes.OR_HOR) {
                                prop = sb.getStdsObj().getAvgArmW(sb.getOrient());
                            } else {
                                prop = sb.getStdsObj().getAvgArmH(sb.getOrient());
                            }
                        }
                        an.getObject().get(ind).addBndbox(bn, 1, prop);
                        System.out.printf("\tFound %d ID:%s NOMBRE:%s \t (score: %.4f) \t (xmin: %d \t ymin: %d \t xmax: %d \t ymax: %d)\n",
                                (int) classes[i],
                                an.getObject().get(ind).getName(),
                                an.getObject().get(ind).getLabel(),
                                bn.getScore(),
                                (int) (boxes[i][1] * imageSize[0]),
                                (int) (boxes[i][0] * imageSize[1]),
                                (int) (boxes[i][3] * imageSize[0]),
                                (int) (boxes[i][2] * imageSize[1]));
                    }
                }
                if (!foundSomething) {
                    System.out.println("TSFLOW:NO ENCONTRO OBJECCTOS CON PROBABILIDAD >  " + model.getMinScore());
                }

                res.add(an);

            } catch (Exception e) {
                System.out.println(e);
            } finally {

            }
        }
        return res;
    }

    @Override
    public Annotation detectObject(ModModel model, byte data[], int alto, int ancho) {

        Annotation an = new Annotation();

        //final String filename = "/home/jcgarciaca/data/java_tf/source/DetectObjects/images/galletas/image1.JPEG";
        int[] imageSize = {ancho, alto};
        // //          System.out.println("width: " + imageSize[0] + " height: " + imageSize[1]);
        List<Tensor<?>> outputs = null;
        try (Tensor<UInt8> input = makeImageTensor(data, alto, ancho)) {

            outputs
                    = ((SavedModelBundle) model.getWeightModel())
                            .session()
                            .runner()
                            .feed("image_tensor", input)
                            .fetch("detection_scores")
                            .fetch("detection_classes")
                            .fetch("detection_boxes")
                            .run();
        } catch (IOException ex) {
            Logger.getLogger(SRCTSFlow.class.getName()).log(Level.SEVERE, null, ex);
        }
        try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
                Tensor<Float> classesT = outputs.get(1).expect(Float.class);
                Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
            // All these tensors have:
            // - 1 as the first dimension
            // - maxObjects as the second dimension
            // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
            // This can be verified by looking at scoresT.shape() etc.
            int maxObjects = (int) scoresT.shape()[1];
            //          System.out.println(maxObjects);
            float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
            float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
            float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
            // Print all objects whose score is at least 0.5.
            //          System.out.printf("* %s\n", "CONF:");
            boolean foundSomething = false;
            for (int i = 0; i < scores.length; ++i) {

                double score = update(scores[i]);
                if (score > model.getMinScore()) {
                    continue;
                }
                foundSomething = true;
                String cn = model.getItemName((int) classes[i]);
                int ind = an.getIndxObj(cn);

                if (ind < 0) {
                    Objeto ob = new Objeto();
                    ob.setName(cn);
                    an.getObject().add(ob);
                    ind = an.getObject().size() - 1;
                }
                Bndbox bn = new Bndbox();

                bn.setScore(score);
                bn.setId("T:" + cn);
                bn.setXmin((int) (boxes[i][1] * imageSize[0]));
                bn.setYmin((int) (boxes[i][0] * imageSize[1]));
                bn.setXmax((int) (boxes[i][3] * imageSize[0]));
                bn.setYmax((int) (boxes[i][2] * imageSize[1]));

                an.getObject().get(ind).addBndbox(bn, 1, 1);
                System.out.printf("\tFound %d ID:%s NOMBRE:%s \t (score: %.4f) \t (xmin: %d \t ymin: %d \t xmax: %d \t ymax: %d)\n",
                        (int) classes[i], update(scores[i]),
                        an.getObject().get(ind).getName(),
                        an.getObject().get(ind).getLabel(),
                        (int) (boxes[i][1] * imageSize[0]),
                        (int) (boxes[i][0] * imageSize[1]),
                        (int) (boxes[i][3] * imageSize[0]),
                        (int) (boxes[i][2] * imageSize[1]));
            }
            if (!foundSomething) {
                //          System.out.println("No objects detected with a high enough score.");
            }
        }

        return an;
    }

    private double update(double score) {
        // TODO Auto-generated method stub
//        double x1 = 0.07, y1 = 0.5, x2 = 0.9, y2 = 0.9;
//        double m = (double) ((y2 - y1) / (x2 - x1));
//        double b = (double) (y2 - (m * x2));
//        double n_score = (double) ((m * score) + b);
        return score;
    }

    private int[] getImageSize(String filename) {
        // TODO Auto-generated method stub
        int[] size = new int[2];
        try {
            Image picture = ImageIO.read(new File(filename));
            size[0] = picture.getWidth(null);
            size[1] = picture.getHeight(null);
            picture.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return size;
    }

    @Override
    public String detectFeature(ModModel model, Mat face, String opts) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String detectFeature(Net n, List<item> itms, Mat face, String opts) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
