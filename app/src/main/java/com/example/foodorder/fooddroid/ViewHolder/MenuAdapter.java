package com.example.foodorder.fooddroid.ViewHolder;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.drawable.Drawable;
import android.support.annotation.Nullable;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.bumptech.glide.load.resource.drawable.GlideDrawable;
import com.bumptech.glide.request.RequestListener;

import com.example.foodorder.fooddroid.Common.Common;
import com.example.foodorder.fooddroid.Interface.ItemClickListener;
import com.example.foodorder.fooddroid.Model.Category;
import com.example.foodorder.fooddroid.R;

import com.squareup.picasso.Picasso;
import com.squareup.picasso.Target;



import java.util.List;

public class MenuAdapter extends RecyclerView.Adapter<MenuAdapter.MenuViewHolder> {


    private List<Category> menuList;
    private Context parentContext;

    public class MenuViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener{
        private ImageView imageView;
        private TextView textView;
        private ItemClickListener itemClickListener;
        private TextView menu_desc;
        //final ProgressBar progressBar ;
        public MenuViewHolder(View itemView) {
            super(itemView);

            textView=itemView.findViewById(R.id.menu_name);
            menu_desc=itemView.findViewById(R.id.menu_desc);
            imageView=itemView.findViewById(R.id.menu_image);
            itemView.setOnClickListener(this);
        }

        public void setItemClickListener(ItemClickListener itemClickListener){
            this.itemClickListener=itemClickListener;
        }


        @Override
        public void onClick(View v) {
            itemClickListener.OnClick(v,getAdapterPosition(),false);

        }
    }

    public MenuAdapter(List<Category> menuList,Context parentContext) {
        this.menuList = menuList;
        this.parentContext=parentContext;
    }

    @Override
    public MenuViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.menu_item, parent, false);

        return new MenuViewHolder(itemView);
    }

    @Override
    public void onBindViewHolder(final MenuViewHolder holder, int position) {

        Category menu = menuList.get(position);
        holder.textView.setText(menu.getName());
        holder.textView.setTypeface(Common.getTypeFace(parentContext));
        holder.menu_desc.setTypeface(Common.getTypeFace(parentContext));
        Picasso.with(parentContext).load(menu.getImage()).into(holder.imageView);




    }

    @Override
    public int getItemCount() {
        return menuList.size();
    }
}
