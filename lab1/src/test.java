import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class test {
    final static int TRAINSIZE = 450;

    public static void main(String[] args) throws IOException {
//        String source = "D:\\PJ\\IntelliLabs\\Lab1part2\\数据";
//        String dst = "D:\\PJ\\IntelliLabs\\Lab1part2\\train";
//        String testdst = "D:\\PJ\\IntelliLabs\\Lab1part2\\test";
//        for (int i = 1; i <= 12; i++) {
//            String folderpath = source + "/" + i;
//            String dstpath = dst+"/"+i;
//            String testpath = testdst+"/"+i;
//            File folder = new File(folderpath);
//            if (!folder.exists()) {
//                folder.mkdir();
//            }
////            for (int j = 1; j <= TRAINSIZE; j++) {
////                copyFile(new File(folderpath+"/"+j+".bmp"),dstpath);
////            }
//
//            for (int j = TRAINSIZE+1; j <= 620; j++) {
//                copyFile(new File(folderpath+"/"+j+".bmp"),testpath);
//            }
//
//
//        }


        for (int i = 1; i <= 1800; i++) {

            File file = new File("test/1/" + i + ".bmp");

            file.renameTo(new File("test/1/" + to4(i) + ".bmp"));

        }
    }


    public static String to4(int i) {
        if (i >= 1 && i <= 9) {
            return "000" + i;
        } else if (i >= 10 && i <= 99) {
            return "00" + i;
        } else if (i >= 100 && i <= 999) {
            return "0" + i;
        } else {
            return "" + i;
        }
    }

    public static void copyFile(File source, String dest) throws IOException {
        //创建目的地文件夹
        File destfile = new File(dest);
        if (!destfile.exists()) {
            destfile.mkdir();
        }
        //如果source是文件夹，则在目的地址中创建新的文件夹
        if (source.isDirectory()) {
            File file = new File(dest + "\\" + source.getName());//用目的地址加上source的文件夹名称，创建新的文件夹
            file.mkdir();
            //得到source文件夹的所有文件及目录
            File[] files = source.listFiles();
            if (files.length == 0) {
                return;
            } else {
                for (int i = 0; i < files.length; i++) {
                    copyFile(files[i], file.getPath());
                }
            }

        }
        //source是文件，则用字节输入输出流复制文件
        else if (source.isFile()) {
            FileInputStream fis = new FileInputStream(source);
            //创建新的文件，保存复制内容，文件名称与源文件名称一致
            File dfile = new File(dest + "\\" + source.getName());
            if (!dfile.exists()) {
                dfile.createNewFile();
            }

            FileOutputStream fos = new FileOutputStream(dfile);
            // 读写数据
            // 定义数组
            byte[] b = new byte[1024];
            // 定义长度
            int len;
            // 循环读取
            while ((len = fis.read(b)) != -1) {
                // 写出数据
                fos.write(b, 0, len);
            }

            //关闭资源
            fos.close();
            fis.close();

        }
    }
}
